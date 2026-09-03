"use strict";

const assert = require("node:assert/strict");
const EventEmitter = require("node:events");
const test = require("node:test");

const {
  AckTracker,
  SentBodyReservations,
  contactIdentityIds,
  createCompatibleClientClass,
  installMessageKeyCompatibility,
  isSelfChat,
  normalizeId,
  normalizeMessage,
  normalizeReaction,
  quotedMessageContext,
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

test("normalizes WhatsApp reaction notifications", () => {
  assert.deepEqual(
    normalizeReaction(
      {
        msgId: { fromMe: true, remote: { user: "123", server: "g.us" }, id: "ABC" },
        senderId: { _serialized: "operator@lid" },
        reaction: "👍",
        timestamp: 123,
      },
      ["self@c.us", "self@lid"],
    ),
    {
      chat_id: "123@g.us",
      chatId: "123@g.us",
      message_id: "true_123@g.us_ABC",
      messageId: "true_123@g.us_ABC",
      sender_id: "operator@lid",
      senderId: "operator@lid",
      emoji: "👍",
      reaction: "👍",
      from_self: false,
      fromSelf: false,
      self_chat: false,
      selfChat: false,
      chat_type: "group",
      timestamp: 123,
    },
  );
});

test("derives reaction chats from serialized-only message keys", () => {
  for (const msgId of [
    { $1: "false_chat@lid_FIRST" },
    { _serialized: "false_chat@lid_SECOND" },
  ]) {
    const normalized = normalizeReaction(
      { msgId, senderId: "operator@lid", reaction: "👍" },
      ["self@c.us"],
    );

    assert.ok(normalized);
    assert.equal(normalized.chatId, "chat@lid");
    assert.equal(normalized.messageId, serializedId(msgId));
  }
});

test("marks paired-account reactions in the self-chat", () => {
  const normalized = normalizeReaction(
    {
      msgId: { fromMe: true, remote: "self@lid", id: "ABC" },
      senderId: "self@lid",
      reaction: "✅",
    },
    ["self@c.us", "self@lid"],
  );

  assert.equal(normalized.senderId, "self@c.us");
  assert.equal(normalized.fromSelf, true);
  assert.equal(normalized.selfChat, true);
});

test("rejects removed, malformed, and status reactions", () => {
  assert.equal(
    normalizeReaction(
      { msgId: { remote: "chat@lid", id: "ABC" }, senderId: "user@lid", reaction: "" },
      [],
    ),
    null,
  );
  assert.equal(normalizeReaction({ reaction: "👍" }, []), null);
  assert.equal(
    normalizeReaction(
      {
        msgId: { remote: "status@broadcast", id: "ABC" },
        senderId: "user@lid",
        reaction: "👍",
      },
      [],
    ),
    null,
  );
  assert.equal(
    normalizeReaction(
      {
        msgId: { _serialized: "false_status@broadcast_ABC" },
        senderId: "user@lid",
        reaction: "👍",
      },
      [],
    ),
    null,
  );
});

test("classifies messages without quoted context", async () => {
  assert.deepEqual(await quotedMessageContext({ hasQuotedMsg: false }), {
    participant: null,
    messageId: null,
    status: "not_reply",
  });
});

test("resolves quoted message context", async () => {
  const context = await quotedMessageContext({
    hasQuotedMsg: true,
    getQuotedMessage: async () => ({
      author: { user: "quoted-user", server: "lid" },
      id: { fromMe: false, remote: "chat@lid", id: "ABC" },
    }),
  });

  assert.deepEqual(context, {
    participant: "quoted-user@lid",
    messageId: "false_chat@lid_ABC",
    status: "resolved",
  });
});

test("reports quoted message lookup failures", async () => {
  const context = await quotedMessageContext({
    hasQuotedMsg: true,
    getQuotedMessage: async () => {
      throw new Error("private provider failure");
    },
  });

  assert.deepEqual(context, {
    participant: null,
    messageId: null,
    status: "lookup_failed",
  });
});

test("collects the paired account phone and LID aliases", () => {
  assert.deepEqual(
    contactIdentityIds({ _serialized: "123@c.us" }, [
      { pn: "123@c.us", lid: "456@lid" },
    ]),
    ["123@c.us", "456@lid"],
  );
});

test("recognizes only outbound messages addressed to the paired account as self-chat", () => {
  const ownIds = [{ _serialized: "123@c.us" }, { _serialized: "456@lid" }];

  assert.equal(isSelfChat(true, "123@c.us", ownIds), true);
  assert.equal(isSelfChat(true, "456@lid", ownIds), true);
  assert.equal(isSelfChat(true, "789@c.us", ownIds), false);
  assert.equal(isSelfChat(false, "123@c.us", ownIds), false);
  assert.equal(isSelfChat(true, "123@c.us", []), false);
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
