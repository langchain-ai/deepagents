"use strict";

const assert = require("node:assert/strict");
const EventEmitter = require("node:events");
const test = require("node:test");

const {
  AckTracker,
  SentBodyReservations,
  createCompatibleClientClass,
  installMessageKeyCompatibility,
  normalizeId,
  normalizeMessage,
  serializedId,
  widString,
} = require("../../../deepagents_talon/channels/whatsapp_bridge/id_compat");

test("reads legacy, renamed, and component WhatsApp IDs", () => {
  assert.equal(widString({ _serialized: "123@lid" }), "123@lid");
  assert.equal(widString({ $1: "123@lid" }), "123@lid");
  assert.equal(widString({ user: "123", server: "lid" }), "123@lid");
  assert.equal(
    serializedId({ fromMe: true, remote: { user: "123", server: "lid" }, id: "ABC" }),
    "true_123@lid_ABC",
  );
});

test("reconstructs self state only from boolean true", () => {
  assert.equal(
    serializedId({ fromMe: "false", remote: "123@lid", id: "ABC" }),
    "false_123@lid_ABC",
  );
});

test("normalizes frozen IDs and message fields", () => {
  const id = Object.freeze({ fromMe: false, remote: "123@lid", id: "ABC" });
  const message = normalizeMessage({
    id,
    _data: {
      id,
      from: { $1: "123@lid" },
      to: { user: "456", server: "lid" },
      author: { _serialized: "789@lid" },
    },
  });
  assert.equal(normalizeId(id)._serialized, "false_123@lid_ABC");
  assert.equal(message.id._serialized, "false_123@lid_ABC");
  assert.equal(message.from, "123@lid");
  assert.equal(message.to, "456@lid");
  assert.equal(message.author, "789@lid");
});

test("installs a message-key prototype fallback", () => {
  class MessageKey {}
  const sample = new MessageKey();
  sample.$1 = "false_123@lid_ABC";
  const originalWindow = global.window;
  global.window = {
    require(name) {
      if (name === "WAWebMsgKey") {
        return MessageKey;
      }
      return { Chat: { getModelsArray: () => [{ lastReceivedKey: sample }] } };
    },
  };
  try {
    assert.deepEqual(installMessageKeyCompatibility(), { installed: true, compatible: true });
    assert.equal(sample._serialized, "false_123@lid_ABC");
  } finally {
    global.window = originalWindow;
  }
});

test("installs compatibility before upstream event listeners", async () => {
  const order = [];
  class BaseClient extends EventEmitter {
    constructor() {
      super();
      this.pupPage = {
        evaluate: async () => {
          order.push("compatibility");
          return { installed: true, compatible: true };
        },
      };
    }

    async attachEventListeners() {
      order.push("listeners");
    }
  }
  const CompatibleClient = createCompatibleClientClass(BaseClient);
  const client = new CompatibleClient();
  await client.attachEventListeners();
  assert.deepEqual(order, ["compatibility", "listeners"]);
  assert.deepEqual(client.idCompatibility, { installed: true, compatible: true });
});

test("reconciles an acknowledgement received before send registration", () => {
  const acknowledgements = [];
  const tracker = new AckTracker({
    timeoutMs: 100,
    onAck: (ack, tracked) => acknowledgements.push([ack, tracked]),
    onTimeout: () => assert.fail("unexpected acknowledgement timeout"),
  });
  assert.equal(tracker.record("message", 1), false);
  tracker.register("message", 0);
  assert.deepEqual(acknowledgements, [[1, true]]);
  assert.equal(tracker.pending.size, 0);
  tracker.forget("message");
});

test("tracks terminal acknowledgements after send registration", () => {
  const acknowledgements = [];
  const tracker = new AckTracker({
    timeoutMs: 100,
    onAck: (ack, tracked) => acknowledgements.push([ack, tracked]),
    onTimeout: () => assert.fail("unexpected acknowledgement timeout"),
  });
  tracker.register("message", 0);
  assert.equal(tracker.record("message", 1), true);
  assert.deepEqual(acknowledgements, [[1, true]]);
  assert.equal(tracker.pending.size, 0);
  tracker.forget("message");
});

test("times out sends without an acknowledgement", async () => {
  let resolveTimeout;
  const timedOut = new Promise((resolve) => {
    resolveTimeout = resolve;
  });
  const tracker = new AckTracker({
    timeoutMs: 1,
    onAck: () => assert.fail("unexpected acknowledgement"),
    onTimeout: resolveTimeout,
  });
  const keepAlive = setTimeout(() => {}, 100);
  try {
    tracker.register("message", 0);
    await timedOut;
    assert.equal(tracker.pending.size, 0);
    tracker.forget("message");
  } finally {
    clearTimeout(keepAlive);
  }
});

test("reserves identical sent bodies only while sends are in flight", () => {
  const reservations = new SentBodyReservations();
  const releaseFirst = reservations.reserve("same body");
  const releaseSecond = reservations.reserve("same body");
  assert.equal(reservations.has("same body"), true);
  releaseFirst();
  assert.equal(reservations.has("same body"), true);
  releaseSecond();
  assert.equal(reservations.has("same body"), false);
});
