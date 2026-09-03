"use strict";

function widString(value) {
  if (!value) {
    return null;
  }
  if (typeof value === "string") {
    return value;
  }
  if (typeof value !== "object") {
    return null;
  }
  if (typeof value._serialized === "string" && value._serialized) {
    return value._serialized;
  }
  if (typeof value.$1 === "string" && value.$1) {
    return value.$1;
  }
  if (typeof value.user === "string" && typeof value.server === "string") {
    return `${value.user}@${value.server}`;
  }
  return null;
}

function contactIdentityIds(primary, mappings = []) {
  const aliases = mappings.flatMap((mapping) => [mapping && mapping.lid, mapping && mapping.pn]);
  return [...new Set([primary, ...aliases].map(widString).filter(Boolean))];
}

function isSelfChat(fromSelf, chatId, ownIds) {
  if (fromSelf !== true) {
    return false;
  }
  const serializedChatId = widString(chatId);
  return Boolean(serializedChatId && ownIds.some((value) => widString(value) === serializedChatId));
}

function serializedId(value) {
  const serialized = widString(value);
  if (serialized || !value || typeof value !== "object") {
    return serialized;
  }
  const remote = widString(value.remote);
  if (remote && typeof value.id === "string" && value.id) {
    return `${value.fromMe === true ? "true" : "false"}_${remote}_${value.id}`;
  }
  return typeof value.id === "string" && value.id ? value.id : null;
}

async function quotedMessageContext(message) {
  if (!message.hasQuotedMsg) {
    return { participant: null, messageId: null, status: "not_reply" };
  }
  try {
    const quoted = await message.getQuotedMessage();
    return {
      participant: widString(quoted.author) || widString(quoted.from),
      messageId: serializedId(quoted.id),
      status: "resolved",
    };
  } catch (_error) {
    return { participant: null, messageId: null, status: "lookup_failed" };
  }
}

function normalizeId(value) {
  if (!value || typeof value !== "object" || value._serialized) {
    return value;
  }
  const serialized = serializedId(value);
  if (!serialized) {
    return value;
  }
  try {
    value._serialized = serialized;
    if (value._serialized === serialized) {
      return value;
    }
  } catch (_error) {
    return { ...value, _serialized: serialized };
  }
  return { ...value, _serialized: serialized };
}

function normalizeMessage(message) {
  if (!message || typeof message !== "object") {
    return message;
  }
  const data = message._data && typeof message._data === "object" ? message._data : {};
  message.id = normalizeId(message.id || data.id);
  message._data = { ...data, id: normalizeId(data.id || message.id) };
  message.from = widString(message.from) || widString(data.from);
  message.to = widString(message.to) || widString(data.to);
  message.author = widString(message.author) || widString(data.author);
  return message;
}

function installMessageKeyCompatibility() {
  const MessageKey = window.require("WAWebMsgKey");
  if (!MessageKey || !MessageKey.prototype) {
    return { installed: false, compatible: false };
  }
  const prototype = MessageKey.prototype;
  let descriptorOwner = prototype;
  let descriptor;
  while (descriptorOwner && !descriptor) {
    descriptor = Object.getOwnPropertyDescriptor(descriptorOwner, "_serialized");
    descriptorOwner = Object.getPrototypeOf(descriptorOwner);
  }
  let installed = false;
  if (!descriptor) {
    Object.defineProperty(prototype, "_serialized", {
      configurable: true,
      get() {
        if (typeof this.$1 === "string" && this.$1) {
          return this.$1;
        }
        const remote =
          typeof this.remote === "string"
            ? this.remote
            : this.remote &&
                (this.remote._serialized ||
                  this.remote.$1 ||
                  (this.remote.user && this.remote.server
                    ? `${this.remote.user}@${this.remote.server}`
                    : null));
        if (!remote || typeof this.id !== "string" || !this.id) {
          return undefined;
        }
        return `${this.fromMe === true ? "true" : "false"}_${remote}_${this.id}`;
      },
    });
    installed = true;
  }
  const chats = window.require("WAWebCollections").Chat.getModelsArray();
  const sample = chats.map((chat) => chat.lastReceivedKey).find(Boolean);
  const compatible = !sample || typeof sample._serialized === "string";
  return { installed, compatible };
}

async function installPageCompatibility(page) {
  const result = await page.evaluate(installMessageKeyCompatibility);
  if (!result || result.compatible !== true) {
    throw new Error("WhatsApp message key compatibility check failed");
  }
  return result;
}

function createCompatibleClientClass(ClientClass) {
  return class CompatibleClient extends ClientClass {
    async attachEventListeners() {
      try {
        this.idCompatibility = await installPageCompatibility(this.pupPage);
      } catch (error) {
        this.emit("compatibility_error", error);
        throw error;
      }
      return super.attachEventListeners();
    }
  };
}

function isTerminalAck(ack) {
  return Number.isFinite(ack) && (ack < 0 || ack >= 1);
}

class AckTracker {
  constructor({ timeoutMs, onAck, onTimeout, maxEarlyAcks = 200 }) {
    this.timeoutMs = timeoutMs;
    this.onAck = onAck;
    this.onTimeout = onTimeout;
    this.maxEarlyAcks = maxEarlyAcks;
    this.known = new Set();
    this.pending = new Map();
    this.early = new Map();
  }

  register(id, currentAck) {
    this.known.add(id);
    const pendingTimer = this.pending.get(id);
    if (pendingTimer) {
      clearTimeout(pendingTimer);
      this.pending.delete(id);
    }
    const earlyAck = this.early.get(id);
    this.early.delete(id);
    const ack = isTerminalAck(Number(currentAck)) ? Number(currentAck) : earlyAck;
    if (isTerminalAck(ack)) {
      this.onAck(ack, true);
      return;
    }
    const timer = setTimeout(() => {
      this.pending.delete(id);
      this.onTimeout();
    }, this.timeoutMs);
    timer.unref();
    this.pending.set(id, timer);
  }

  record(id, value) {
    const ack = Number(value);
    if (!id || !Number.isFinite(ack)) {
      return false;
    }
    if (!this.known.has(id)) {
      if (isTerminalAck(ack)) {
        this.rememberEarly(id, ack);
      }
      return false;
    }
    const timer = this.pending.get(id);
    if (timer && isTerminalAck(ack)) {
      clearTimeout(timer);
      this.pending.delete(id);
    }
    this.onAck(ack, Boolean(timer));
    return true;
  }

  forget(id) {
    this.known.delete(id);
    const timer = this.pending.get(id);
    if (timer) {
      clearTimeout(timer);
      this.pending.delete(id);
    }
    this.early.delete(id);
  }

  rememberEarly(id, ack) {
    this.early.set(id, ack);
    while (this.early.size > this.maxEarlyAcks) {
      this.early.delete(this.early.keys().next().value);
    }
  }
}

class SentBodyReservations {
  constructor() {
    this.counts = new Map();
  }

  reserve(body) {
    if (!body) {
      return () => {};
    }
    const key = String(body);
    this.counts.set(key, (this.counts.get(key) || 0) + 1);
    return () => this.release(key);
  }

  has(body) {
    return Boolean(body && this.counts.has(String(body)));
  }

  release(key) {
    const count = this.counts.get(key) || 0;
    if (count <= 1) {
      this.counts.delete(key);
    } else {
      this.counts.set(key, count - 1);
    }
  }
}

module.exports = {
  AckTracker,
  SentBodyReservations,
  contactIdentityIds,
  createCompatibleClientClass,
  installMessageKeyCompatibility,
  installPageCompatibility,
  isSelfChat,
  normalizeId,
  normalizeMessage,
  quotedMessageContext,
  serializedId,
  widString,
};
