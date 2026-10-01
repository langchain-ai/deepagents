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

function harness(download) {
  class Client extends EventEmitter {
    initialize() {}
  }
  const context = vm.createContext({
    Buffer, URL, setTimeout, clearTimeout,
    console: { log() {}, error() {} },
    process: { env: { WHATSAPP_BRIDGE_TOKEN: "test" }, cwd: () => "/tmp", on() {} },
    require(name) {
      if (name === "fs") return { mkdirSync() {}, readdirSync: () => [], existsSync: () => false, writeFileSync() {} };
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
    enqueue: vm.runInContext("enqueueMessage", context),
    queue: vm.runInContext("queue", context),
    advance(ms) { vm.runInContext(`Date.now = () => ${Date.now() + ms}`, context); },
    async request(method, url, body = {}) {
      const req = new EventEmitter();
      Object.assign(req, { method, url, headers: { authorization: "Bearer test" } });
      let status;
      let payload;
      const res = { writeHead(code) { status = code; }, end(data) { payload = JSON.parse(data.toString()); } };
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

test("deferred requests validate identity and remain retryable until release", async () => {
  let downloads = 0;
  const bridge = harness(async () => { downloads += 1; return null; });
  await bridge.enqueue(message("older", "attachment", true), false);
  const entry = bridge.queue[0];
  assert.equal((await bridge.request("POST", "/prepare", { ...entry, chat_id: "wrong" })).status, 404);
  assert.equal(downloads, 0);
  assert.equal((await bridge.request("POST", "/prepare", entry)).status, 200);
  assert.equal((await bridge.request("POST", "/prepare", entry)).status, 200);
  assert.equal(downloads, 1);
  await bridge.request("POST", "/release", { inputs: [{ ...entry, chat_id: "wrong" }] });
  assert.equal((await bridge.request("POST", "/prepare", entry)).status, 200);
  await bridge.request("POST", "/release", { inputs: [entry] });
  assert.equal((await bridge.request("POST", "/prepare", entry)).status, 404);
});

test("envelope overload is bounded before media or context work starts", async () => {
  const bridge = harness(async () => assert.fail("unexpected download"));
  for (let index = 0; index < 200; index += 1) {
    await bridge.enqueue(message(String(index), "attachment", true), false);
  }
  assert.equal(bridge.queue.length, 128);
});

test("concurrent retries share preparation even after the original response is lost", async () => {
  const entered = barrier();
  const release = barrier();
  let downloads = 0;
  const bridge = harness(async () => {
    downloads += 1;
    entered.release();
    await release.promise;
    return { data: Buffer.from("test").toString("base64"), mimetype: "image/png" };
  });
  await bridge.enqueue(message("slow", "attachment", true), false);
  const [entry] = (await bridge.request("GET", "/messages")).payload;
  const lost = bridge.request("POST", "/prepare", entry);
  await entered.promise;
  bridge.advance(180000);
  const retry = bridge.request("POST", "/prepare", entry);
  release.release();
  await lost;
  const result = await retry;
  assert.equal(result.status, 200);
  assert.equal(result.payload.media_paths.length, 1);
  assert.deepEqual(await bridge.request("POST", "/prepare", entry), result);
  assert.equal(downloads, 1);
});

test("releasing rejected envelopes restores capacity without downloading media", async () => {
  const bridge = harness(async () => assert.fail("unexpected download"));
  for (let index = 0; index < 128; index += 1) {
    await bridge.enqueue(message(String(index), "rejected", true), false);
  }
  const inputs = (await bridge.request("GET", "/messages")).payload;
  assert.equal((await bridge.request("POST", "/release", { inputs })).status, 200);
  assert.equal((await bridge.request("POST", "/release", { inputs })).status, 200);
  await bridge.enqueue(message("legitimate", "hello"), false);
  assert.equal((await bridge.request("GET", "/messages")).payload[0].message_id, "legitimate");
});

test("delivered inputs survive the TTL while unclaimed inputs expire", async () => {
  const bridge = harness(async () => null);
  await bridge.enqueue(message("waiting", "queued behind transcription", true), false);
  const [claimed] = (await bridge.request("GET", "/messages")).payload;
  await bridge.enqueue(message("unclaimed", "abandoned", true), false);
  bridge.advance(180000);
  assert.deepEqual((await bridge.request("GET", "/messages")).payload, []);
  assert.equal((await bridge.request("POST", "/prepare", claimed)).status, 200);
});
