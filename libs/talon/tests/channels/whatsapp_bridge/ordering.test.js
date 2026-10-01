"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const EventEmitter = require("node:events");
const test = require("node:test");

const bridgeDir = path.resolve(__dirname, "../../../deepagents_talon/channels/whatsapp_bridge");

function barrier() {
  let release;
  const promise = new Promise((resolve) => { release = resolve; });
  return { promise, release };
}

function harness(download, { storedBytes = 0, failWrite = false } = {}) {
  const files = new Map();
  const filesystem = {
    mkdirSync() {}, existsSync: () => false,
    readdirSync: (dir) => dir.endsWith("/inbound")
      ? [...files.keys()].map((file) => ({ name: path.basename(file), isFile: () => true })) : [],
    statSync: (file) => files.get(file),
    unlinkSync: (file) => files.delete(file),
    writeFileSync(file, data) {
      files.set(file, { size: data.length, mtimeMs: Date.now() });
      if (failWrite) throw new Error("disk full");
    },
  };
  if (storedBytes) files.set("/tmp/media/inbound/existing", { size: storedBytes, mtimeMs: Date.now() });
  class Client extends EventEmitter {
    initialize() {}
  }
  const context = vm.createContext({
    Buffer, URL, setTimeout, clearTimeout,
    setInterval: () => ({ unref() {} }),
    console: { log() {}, error() {} },
    process: { env: { WHATSAPP_BRIDGE_TOKEN: "test" }, cwd: () => "/tmp", on() {} },
    require(name) {
      if (name === "fs") return filesystem;
      if (name === "http") return { createServer: () => ({ listen() {} }) };
      if (name === "whatsapp-web.js") return { Client, LocalAuth: class {}, MessageMedia: class {} };
      if (name === "qrcode-terminal") return {};
      if (name === "./media") return { downloadMedia: download };
      if (name.startsWith("./")) return require(path.join(bridgeDir, name));
      return require(name);
    },
  });
  vm.runInContext(fs.readFileSync(path.join(bridgeDir, "bridge.js"), "utf8"), context);
  vm.runInContext('status = "connected";', context);
  return {
    files,
    pending: vm.runInContext("pendingMessages", context),
    enqueue: vm.runInContext("enqueueMessage", context),
    queue: vm.runInContext("queue", context),
    async request(method, url, body = {}) {
      const req = new EventEmitter();
      Object.assign(req, { method, url, headers: { authorization: "Bearer test" } });
      let status;
      let payload;
      const res = new EventEmitter();
      Object.assign(res, { writeHead(code) { status = code; }, end(data) { payload = JSON.parse(data.toString()); } });
      const handling = vm.runInContext("handle", context)(req, res);
      req.emit("data", Buffer.from(JSON.stringify(body)));
      req.emit("end");
      await handling;
      return { status, payload };
    },
  };
}

function message(id, body, hasMedia = false) {
  return {
    id: { _serialized: id }, from: "sender@c.us", to: "bot@c.us", body,
    hasMedia, type: hasMedia ? "image" : "chat", _data: { size: 4 },
    getChat: async () => ({ name: "chat" }), getContact: async () => ({ name: "sender" }),
  };
}

test("envelopes are ordered before downloads and stop is available during stalled media", async () => {
  const entered = barrier();
  const release = barrier();
  let downloads = 0;
  const bridge = harness(async () => {
    downloads += 1;
    entered.release();
    await release.promise;
    return { data: Buffer.from("test").toString("base64"), mimetype: "image/png" };
  });
  await bridge.enqueue(message("older", "attachment", true), false);
  await bridge.enqueue(message("stop", "/stop"), false);
  const envelopes = (await bridge.request("GET", "/messages")).payload;
  assert.deepEqual(envelopes.map((entry) => entry.message_id), ["older", "stop"]);
  assert.equal(downloads, 0);
  const first = envelopes[0];
  const preparation = bridge.request("POST", "/prepare", first);
  await entered.promise;
  await bridge.enqueue(message("later", "/stop"), false);
  const later = await bridge.request("GET", "/messages");
  assert.equal(later.payload[0].text, "/stop");
  release.release();
  assert.equal((await preparation).status, 200);
  assert.deepEqual((await bridge.request("GET", "/messages")).payload, []);
});

test("deferred requests must match the admitted conversation and are single use", async () => {
  let downloads = 0;
  const bridge = harness(async () => { downloads += 1; return null; });
  await bridge.enqueue(message("older", "attachment", true), false);
  const entry = bridge.queue[0];
  assert.equal((await bridge.request("POST", "/prepare", { ...entry, chat_id: "wrong" })).status, 404);
  assert.equal(downloads, 0);
  assert.equal((await bridge.request("POST", "/prepare", entry)).status, 200);
  assert.equal((await bridge.request("POST", "/prepare", entry)).status, 404);
  assert.equal(downloads, 1);
});

test("envelope overload is bounded before media or context work starts", async () => {
  const bridge = harness(async () => assert.fail("unexpected download"));
  for (let index = 0; index < 200; index += 1) {
    await bridge.enqueue(message(String(index), "attachment", true), false);
  }
  assert.equal(bridge.queue.length, 128);
});

const inertMedia = { data: Buffer.from("test").toString("base64"), mimetype: "image/png" };

test("completed attachments release admission capacity but remain discardable", async () => {
  const bridge = harness(async () => inertMedia);
  let completed;
  for (let index = 0; index < 128; index++) {
    await bridge.enqueue(message(String(index), "", true), false);
    [completed] = (await bridge.request("GET", "/messages")).payload;
    assert.equal((await bridge.request("POST", "/prepare", completed)).status, 200);
  }
  assert.equal(bridge.files.size, 128);
  for (let index = 0; index < 128; index++) {
    await bridge.enqueue(message(`text-${index}`, "hello"), false);
    const entries = (await bridge.request("GET", "/messages")).payload;
    assert.deepEqual(entries.map((entry) => entry.message_id), [`text-${index}`]);
  }
  await bridge.enqueue(message("overflow", "hello"), false);
  assert.deepEqual((await bridge.request("GET", "/messages")).payload, []);
  assert.equal((await bridge.request("POST", "/prepare", completed)).status, 404);
  await bridge.request("POST", "/discard", completed);
  assert.equal(bridge.files.size, 127);
});

test("discard prevents downloads and releases pending capacity", async () => {
  const bridge = harness(async () => assert.fail("unexpected download"));
  for (let index = 0; index < 128; index++) await bridge.enqueue(message(String(index), "", true), false);
  const entries = (await bridge.request("GET", "/messages")).payload;
  for (const entry of entries) {
    await bridge.request("POST", "/discard", entry);
    assert.equal((await bridge.request("POST", "/prepare", entry)).status, 404);
  }
  await bridge.enqueue(message("new", "", true), false);
  assert.equal(bridge.queue.length, 1);
  assert.equal(bridge.files.size, 0);
});

test("cancellation during downloads prevents writes and releases concurrency", async () => {
  const release = barrier();
  const entered = barrier();
  const bridge = harness(async () => { entered.release(); await release.promise; return inertMedia; });
  await bridge.enqueue(message("cancel", "", true), false);
  const entry = bridge.queue[0];
  const preparing = bridge.request("POST", "/prepare", entry);
  await entered.promise;
  await bridge.request("POST", "/discard", entry);
  release.release();
  await preparing;
  assert.equal(bridge.files.size, 0);
  await bridge.enqueue(message("next", "", true), false);
  assert.equal((await bridge.request("POST", "/prepare", bridge.queue[1])).status, 200);
  assert.equal(bridge.files.size, 1);
});

test("concurrency never exceeds four downloads", async () => {
  const release = barrier();
  const entered = barrier();
  let downloads = 0;
  const bridge = harness(async () => {
    downloads++;
    if (downloads === 4) entered.release();
    await release.promise;
    return inertMedia;
  });
  for (let index = 0; index < 5; index++) await bridge.enqueue(message(String(index), "", true), false);
  const running = bridge.queue.slice(0, 4).map((entry) => bridge.request("POST", "/prepare", entry));
  await entered.promise;
  assert.equal((await bridge.request("POST", "/prepare", bridge.queue[4])).status, 429);
  assert.equal(downloads, 4);
  release.release();
  await Promise.all(running);
  assert.equal((await bridge.request("POST", "/prepare", bridge.queue[4])).status, 200);
});

test("aggregate storage rejects downloads until expired files are reclaimed", async () => {
  let downloads = 0;
  const bridge = harness(async () => { downloads++; return inertMedia; }, { storedBytes: 256 * 1024 * 1024 });
  await bridge.enqueue(message("full", "", true), false);
  await bridge.request("POST", "/prepare", bridge.queue[0]);
  assert.equal(downloads, 0);
  for (const stat of bridge.files.values()) stat.mtimeMs = 0;
  await bridge.enqueue(message("space", "", true), false);
  await bridge.request("POST", "/prepare", bridge.queue[1]);
  assert.equal(downloads, 1);
  assert.equal(bridge.files.size, 1);
});

test("failed writes remove partial files", async () => {
  const bridge = harness(async () => inertMedia, { failWrite: true });
  await bridge.enqueue(message("partial", "", true), false);
  const result = await bridge.request("POST", "/prepare", bridge.queue[0]);
  assert.deepEqual(result.payload.media_paths, []);
  assert.equal(bridge.files.size, 0);
});

for (const size of [undefined, null, "", -1, Infinity, 65 * 1024 * 1024]) {
  test(`unusable media size ${size} does not download`, async () => {
    const bridge = harness(async () => assert.fail("unexpected download"));
    const input = message("size", "", true);
    input._data = { size };
    await bridge.enqueue(input, false);
    await bridge.request("POST", "/prepare", bridge.queue[0]);
    assert.equal(bridge.files.size, 0);
  });
}

test("simultaneous downloads cannot overfill retained storage", async () => {
  const release = barrier();
  const entered = barrier();
  let downloads = 0;
  const limit = 256 * 1024 * 1024;
  const bridge = harness(async () => {
    if (++downloads === 4) entered.release();
    await release.promise;
    return inertMedia;
  }, { storedBytes: limit - 4 });
  for (let index = 0; index < 4; index++) await bridge.enqueue(message(String(index), "", true), false);
  const running = bridge.queue.map((entry) => bridge.request("POST", "/prepare", entry));
  await entered.promise;
  release.release();
  const results = await Promise.all(running);
  assert.equal(results.filter((result) => result.payload.media_paths.length).length, 1);
  assert.equal([...bridge.files.values()].reduce((total, stat) => total + stat.size, 0), limit);
});

test("missing, invalid, expired and mismatched capabilities cannot prepare media", async () => {
  const bridge = harness(async () => assert.fail("unexpected download"));
  await bridge.enqueue(message("id", "", true), false);
  const entry = bridge.queue[0];
  for (const body of [{}, { ...entry, preparation_token: "invalid" }, { ...entry, message_id: "wrong" }]) {
    assert.equal((await bridge.request("POST", "/prepare", body)).status, 404);
  }
  bridge.pending.get(entry.preparation_token).expires = 0;
  assert.equal((await bridge.request("POST", "/prepare", entry)).status, 404);
  assert.equal(bridge.files.size, 0);
});

test("discard after response removes abandoned attachments", async () => {
  const bridge = harness(async () => inertMedia);
  await bridge.enqueue(message("done", "", true), false);
  const entry = bridge.queue[0];
  await bridge.request("POST", "/prepare", entry);
  assert.equal(bridge.files.size, 1);
  await bridge.request("POST", "/discard", entry);
  assert.equal(bridge.files.size, 0);
});
