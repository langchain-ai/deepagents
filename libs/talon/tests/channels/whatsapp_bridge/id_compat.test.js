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
  messageSenderId,
  normalizeId,
  normalizeMessage,
  reactionEntry,
  quotedMessageContext,
  serializedId,
  widString,
} = require("../../../deepagents_talon/channels/whatsapp_bridge/id_compat");

test("self-chat messages and approval reactions use the same paired-account identity", () => {
  const botId = "phone@c.us";
  const aliases = [botId, "alias@lid"];
  for (const author of aliases) {
    for (const emoji of ["\u{1f44d}", "\u{1f44e}"]) {
      const message = normalizeMessage({ author: { _serialized: author } });
      const sender = messageSenderId(message, true, botId, botId);
      const reaction = reactionEntry({
        id: { fromMe: true },
        msgId: { fromMe: true, remote: "alias@lid", id: "PROMPT" },
        senderId: author,
        reaction: emoji,
      }, botId, aliases);
      assert.equal(sender, reaction.user_id);
      assert.equal(sender, botId);
    }
  }
  assert.equal(messageSenderId({ author: "other@lid" }, false, botId, "group@g.us"), "other@lid");
  assert.equal(messageSenderId({}, false, botId, "other@c.us"), "other@c.us");
});

test("queues reactions with the target message ID and reacting sender", () => {
  const reaction = {
    id: { fromMe: false, remote: "chat@g.us", id: "REACTION" },
    msgId: { fromMe: true, remote: "chat@g.us", id: "PROMPT" },
    senderId: { _serialized: "operator@lid" },
    reaction: "\u{1f44d}",
  };
  assert.deepEqual(reactionEntry(reaction, "bot@c.us", ["bot@c.us"]), {
    event_type: "reaction", chat_id: "chat@g.us", user_id: "operator@lid",
    message_id: "true_chat@g.us_PROMPT", text: "\u{1f44d}",
    from_self: false, self_chat: false,
  });
  assert.equal(reactionEntry({ ...reaction, reaction: "" }, "bot@c.us", []), null);
  assert.equal(reactionEntry({ ...reaction, senderId: null }, "bot@c.us", []), null);
  const own = { ...reaction, id: { ...reaction.id, fromMe: true } };
  assert.equal(reactionEntry(own, "bot@c.us", ["bot@c.us"]), null);
  const self = { ...own, msgId: { ...own.msgId, remote: "bot@lid" } };
  const entry = reactionEntry(self, "bot@c.us", ["bot@c.us", "bot@lid"]);
  assert.equal(entry.user_id, "bot@c.us");
  assert.equal(entry.self_chat, true);
});

test("reaction diagnostics explain filtering without exposing payloads", (t) => {
  const logs = [];
  t.mock.method(console, "log", (line) => logs.push(line));
  const reaction = {
    id: { fromMe: false },
    msgId: { fromMe: true, remote: "private-chat", id: "private-message" },
    senderId: "private-sender", reaction: "private-emoji",
  };
  const cases = [
    [{}, null],
    [{ msgId: null }, "missing_message_id"],
    [{ msgId: { _serialized: "private-message" } }, "missing_chat_id"],
    [{ senderId: null }, "missing_sender_id"],
    [{ reaction: "" }, "missing_emoji"],
    [{ msgId: { remote: "status@broadcast", id: "private-message" } }, "status_broadcast"],
    [{ id: { fromMe: true } }, "self_outside_self_chat"],
  ];
  for (const [changes, reason] of cases) {
    const entry = reactionEntry({ ...reaction, ...changes }, "private-bot", ["private-bot"]);
    assert.equal(entry === null, reason !== null);
    const event = JSON.parse(logs.at(-1).split("talon_event ")[1]);
    assert.equal(event.reason, reason);
    assert.equal(event.event, reason ? "whatsapp.bridge.reaction.rejected" : "whatsapp.bridge.reaction.converted");
  }
  assert.equal(logs.some((line) => line.includes("private-")), false);
});

test("reads legacy, renamed, and component WhatsApp IDs", () => {
  assert.equal(widString({ _serialized: "123@lid" }), "123@lid");
  assert.equal(widString({ $1: "123@lid" }), "123@lid");
  assert.equal(widString({ user: "123", server: "lid" }), "123@lid");
  assert.equal(
    serializedId({ fromMe: true, remote: { user: "123", server: "lid" }, id: "ABC" }),
    "true_123@lid_ABC",
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
