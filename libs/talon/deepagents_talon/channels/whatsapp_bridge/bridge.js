"use strict";

const fs = require("fs");
const http = require("http");
const path = require("path");
const { Client, LocalAuth, MessageMedia } = require("whatsapp-web.js");
const qrcode = require("qrcode-terminal");
const {
  AckTracker,
  SentBodyReservations,
  contactIdentityIds,
  createCompatibleClientClass,
  isSelfChat,
  messageSenderId,
  normalizeMessage,
  quotedMessageContext,
  reactionEntry,
  serializedId,
  widString,
} = require("./id_compat");

const host = process.env.WHATSAPP_BRIDGE_HOST || "127.0.0.1";
const port = Number(process.env.WHATSAPP_BRIDGE_PORT || "3000");
const sessionDir = path.resolve(process.env.WHATSAPP_SESSION_DIR || path.join(process.cwd(), ".whatsapp"));
const mediaDir = path.resolve(process.env.WHATSAPP_MEDIA_DIR || path.join(sessionDir, "..", "media"));
const bridgeToken = process.env.WHATSAPP_BRIDGE_TOKEN || "";
const maxMediaBytes = Number(process.env.WHATSAPP_MAX_MEDIA_BYTES) || 64 * 1024 * 1024;
const webVersionCacheUrl =
  process.env.WHATSAPP_WEB_VERSION_CACHE_URL ||
  "https://raw.githubusercontent.com/wppconnect-team/wa-version/main/html/2.3000.1026029003.html";

const ACK_TIMEOUT_MS = 60 * 1000;
const MAX_CACHED_SENT_MESSAGES = 200;

let status = "disconnected";
let botId = null;
let botIds = [];
let bridgeMediaSends = 0;
const queue = [];
const sentMessageIds = new Set();
const sentMessages = new Map();
const sentBodies = new SentBodyReservations();
const ackTracker = new AckTracker({
  timeoutMs: ACK_TIMEOUT_MS,
  onAck: (ack, tracked) => {
    console.log(`[bridge] Outbound message acknowledgement; ack=${ack} tracked=${tracked}`);
  },
  onTimeout: () => {
    console.error("[bridge] Outbound message acknowledgement timed out");
  },
});

process.on("unhandledRejection", (reason) => {
  const message = reason && reason.message ? reason.message : reason;
  console.error("Unhandled rejection:", message);
});

if (!bridgeToken) {
  console.error("WHATSAPP_BRIDGE_TOKEN is required");
  process.exit(1);
}

fs.mkdirSync(sessionDir, { recursive: true });
fs.mkdirSync(mediaDir, { recursive: true });
cleanStaleLocks(sessionDir);

const chromePath = process.env.CHROME_PATH || process.env.WHATSAPP_CHROME_PATH || findChrome();
if (chromePath) {
  console.log("Using configured Chrome executable");
} else {
  console.log("No system Chrome found; using Puppeteer's bundled browser if available");
}

const puppeteer = {
  headless: true,
  args: ["--no-sandbox", "--disable-setuid-sandbox", "--disable-gpu"],
};
if (chromePath) {
  puppeteer.executablePath = chromePath;
}

const CompatibleClient = createCompatibleClientClass(Client);
const client = new CompatibleClient({
  authStrategy: new LocalAuth({
    dataPath: sessionDir,
  }),
  puppeteer,
  webVersionCache: {
    type: "remote",
    remotePath: webVersionCacheUrl,
  },
});

client.on("qr", (qr) => {
  status = "qr_pending";
  console.log("Scan this QR code to pair WhatsApp:");
  qrcode.generate(qr, { small: true });
});

client.on("ready", () => {
  void onClientReady();
});

client.on("disconnected", (reason) => {
  status = "disconnected";
  console.log(`WhatsApp disconnected: ${reason || "unknown reason"}`);
});

client.on("auth_failure", (message) => {
  status = "disconnected";
  console.error(`WhatsApp auth failure: ${message || "unknown error"}`);
});

client.on("compatibility_error", (error) => {
  status = "compatibility_error";
  console.error(`[bridge] WhatsApp compatibility setup failed: ${error.message || error}`);
});

client.on("message_create", (message) => {
  if (message.fromMe === true) {
    void enqueueMessage(message, true);
  }
});

client.on("message", (message) => {
  void enqueueMessage(message, false);
});

client.on("message_reaction", (reaction) => {
  console.log('[bridge] talon_event {"event":"whatsapp.bridge.reaction.received"}');
  try {
    const entry = reactionEntry(reaction, botId, botIds);
    if (entry) {
      queue.push(entry);
      console.log('[bridge] talon_event {"event":"whatsapp.bridge.reaction.queued"}');
    }
  } catch (error) {
    console.log('[bridge] talon_event {"event":"whatsapp.bridge.reaction.parse_failed"}');
    throw error;
  }
});

client.on("media_uploaded", (message) => {
  if (bridgeMediaSends === 0) {
    void enqueueMessage(message, true);
  }
});

client.on("message_ack", (message, ack) => {
  recordMessageAck(message, ack);
});

async function onClientReady() {
  try {
    const compatibility = client.idCompatibility;
    if (!compatibility || compatibility.compatible !== true) {
      throw new Error("WhatsApp message key compatibility is not active");
    }
    const webVersion = safeVersion(await client.getWWebVersion());
    botIds = await resolveBotIds();
    botId = botIds[0] || null;
    status = "connected";
    console.log(
      `[bridge] WhatsApp connected; webVersion=${webVersion} idCompatibilityInstalled=${compatibility.installed === true} botIdAvailable=${Boolean(botId)} botIdAliasAvailable=${botIds.length > 1}`,
    );
  } catch (error) {
    status = "compatibility_error";
    console.error(`[bridge] WhatsApp compatibility setup failed: ${error.message || error}`);
  }
}

async function resolveBotIds() {
  const initialIds = contactIdentityIds(client.info && client.info.wid);
  if (initialIds.length === 0 || typeof client.getContactLidAndPhone !== "function") {
    return initialIds;
  }
  try {
    const mappings = await client.getContactLidAndPhone(initialIds);
    return contactIdentityIds(initialIds[0], mappings);
  } catch (_error) {
    console.log("[bridge] Paired-account ID aliases unavailable; using primary ID only");
    return initialIds;
  }
}

async function enqueueMessage(message, fromSelf) {
  normalizeMessage(message);
  const data = message && message._data ? message._data : {};
  const messageFrom = widString(message.from) || widString(data.from);
  if (messageFrom === "status@broadcast") {
    return;
  }
  if (fromSelf && isBridgeSentMessage(message)) {
    return;
  }

  const messageId = serializedId(message.id) || serializedId(data.id);
  const messageTo = widString(message.to) || widString(data.to);
  const remoteId = widString(message.id && message.id.remote) || widString(data.id && data.id.remote);
  const chatId = (fromSelf ? messageTo : messageFrom) || remoteId;
  if (!messageId || !chatId) {
    console.error("Skipping WhatsApp message without a message id or chat id");
    return;
  }
  const selfChat = isSelfChat(fromSelf, chatId, botIds);
  if (fromSelf && !selfChat) {
    console.log("[bridge] Ignoring paired-account message outside the self-chat");
    return;
  }

  const [chat, contact] = await Promise.all([safeGetChat(message), safeGetContact(message)]);
  const media = await downloadMessageMedia(message);
  const mediaType = classifyMedia(message, media);
  if (message.hasMedia && media.length === 0) {
    console.log(
      `[bridge] Message media unavailable; type=${message.type || "unknown"} mediaType=${mediaType}`,
    );
  }
  const senderId = messageSenderId(message, fromSelf, botId, messageFrom);
  const senderName =
    (contact && (contact.pushname || contact.name || contact.shortName)) ||
    data.notifyName ||
    data.senderName ||
    senderId ||
    null;
  const chatName = (chat && chat.name) || data.chatName || chatId;
  const isGroup =
    chat && typeof chat.isGroup === "boolean" ? chat.isGroup : chatId.endsWith("@g.us");
  const quote = await quotedMessageContext(message);

  const entry = {
    text: message.body || "",
    body: message.body || "",
    message_type: message.type || "chat",
    messageType: message.type || "chat",
    media_type: mediaType,
    mediaType,
    chat_id: chatId,
    chatId,
    chat_id_from: messageFrom,
    chatIdFrom: messageFrom,
    chat_name: chatName,
    chatName,
    chat_type: isGroup ? "group" : "direct",
    chatType: isGroup ? "group" : "direct",
    isGroup,
    user_id: senderId || null,
    senderId: senderId || null,
    user_name: senderName,
    senderName,
    message_id: messageId,
    messageId,
    has_media: Boolean(message.hasMedia || media.length > 0),
    hasMedia: Boolean(message.hasMedia || media.length > 0),
    media_paths: media.map((item) => item.path),
    mediaPaths: media.map((item) => item.path),
    media_urls: media.map((item) => item.path),
    mediaUrls: media.map((item) => item.path),
    media_mime_types: media.map((item) => item.mimeType),
    mediaMimeTypes: media.map((item) => item.mimeType),
    media_types: media.map((item) => item.mimeType),
    mediaTypes: media.map((item) => item.mimeType),
    media_file_names: media.map((item) => item.fileName),
    from_self: fromSelf,
    fromSelf,
    self_chat: selfChat,
    selfChat,
    mentionedIds: normalizeIds(message.mentionedIds || []),
    botIds,
    quoted_participant: quote.participant,
    quotedParticipant: quote.participant,
    quoted_message_id: quote.messageId,
    quotedMessageId: quote.messageId,
    reply_context_status: quote.status,
    replyContextStatus: quote.status,
    raw_message: {
      from: messageFrom,
      to: messageTo,
      author: widString(message.author),
      fromMe: message.fromMe === true,
      idFromMe: message.id && message.id.fromMe === true,
      timestamp: message.timestamp || null,
    },
  };

  console.log(
    `[bridge] Queued message; fromSelf=${fromSelf} hasMedia=${entry.hasMedia} chatType=${entry.chatType}`,
  );
  queue.push(entry);
}

function cleanStaleLocks(dir) {
  let entries;
  try {
    entries = fs.readdirSync(dir, { withFileTypes: true });
  } catch (_error) {
    return;
  }
  for (const entry of entries) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      cleanStaleLocks(full);
      continue;
    }
    if (/^Singleton(Lock|Socket|Cookie)$/.test(entry.name)) {
      try {
        fs.unlinkSync(full);
      } catch (_error) {
        // Best effort cleanup only.
      }
    }
  }
}

function findChrome() {
  const candidates =
    process.platform === "darwin"
      ? [
          "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
          "/Applications/Chromium.app/Contents/MacOS/Chromium",
          "/Applications/Google Chrome Canary.app/Contents/MacOS/Google Chrome Canary",
        ]
      : process.platform === "win32"
        ? [
            process.env.PROGRAMFILES
              ? path.join(process.env.PROGRAMFILES, "Google", "Chrome", "Application", "chrome.exe")
              : null,
            process.env["PROGRAMFILES(X86)"]
              ? path.join(
                  process.env["PROGRAMFILES(X86)"],
                  "Google",
                  "Chrome",
                  "Application",
                  "chrome.exe",
                )
              : null,
          ]
        : [
            "/usr/bin/google-chrome",
            "/usr/bin/google-chrome-stable",
            "/usr/bin/chromium",
            "/usr/bin/chromium-browser",
            "/snap/bin/chromium",
          ];

  for (const candidate of candidates) {
    if (candidate && fs.existsSync(candidate)) {
      return candidate;
    }
  }
  return undefined;
}

async function safeGetChat(message) {
  try {
    return await message.getChat();
  } catch (_error) {
    console.log("[bridge] getChat failed (non-fatal)");
    return null;
  }
}

async function safeGetContact(message) {
  try {
    return await message.getContact();
  } catch (_error) {
    console.log("[bridge] getContact failed (non-fatal)");
    return null;
  }
}

async function downloadMessageMedia(message) {
  if (!message.hasMedia) {
    return [];
  }
  try {
    const messageId = serializedId(message.id) || String(Date.now());
    const expectedSize = messageMediaSize(message);
    if (expectedSize !== null && expectedSize > maxMediaBytes) {
      console.log(
        `[bridge] Skipping oversized media; bytes=${expectedSize} maxBytes=${maxMediaBytes}`,
      );
      return [];
    }
    const media = await message.downloadMedia();
    if (!media || !media.data) {
      return [];
    }
    const extension = mediaExtension(media.mimetype, message.type);
    const fileName = `${Date.now()}_${messageId.replace(/[^A-Za-z0-9]/g, "_")}.${extension}`;
    const filePath = path.join(mediaDir, fileName);
    const size = decodedBase64Size(media.data);
    if (size > maxMediaBytes) {
      console.log(
        `[bridge] Skipping oversized media; bytes=${size} maxBytes=${maxMediaBytes}`,
      );
      return [];
    }
    fs.writeFileSync(filePath, Buffer.from(media.data, "base64"), { mode: 0o600 });
    return [
      {
        path: filePath,
        mimeType: media.mimetype || "application/octet-stream",
        fileName: media.filename || fileName,
      },
    ];
  } catch (error) {
    console.error("Media download failed:", error.message || error);
    return [];
  }
}

function classifyMedia(message, media) {
  const rawType = String(message.type || "").toLowerCase();
  const mimeType = messageMimeType(message, media);
  if (rawType === "ptt" || rawType === "audio" || mimeType.startsWith("audio/")) {
    return "voice";
  }
  if (rawType === "image" || rawType === "sticker" || mimeType.startsWith("image/")) {
    return "image";
  }
  if (rawType === "video" || mimeType.startsWith("video/")) {
    return "video";
  }
  if (message.hasMedia) {
    return "document";
  }
  return "text";
}

function messageMimeType(message, media) {
  if (media.length > 0) {
    return String(media[0].mimeType || "").toLowerCase();
  }
  const data = message && message._data ? message._data : {};
  return String(message.mimetype || data.mimetype || data.mimetypeOverride || "").toLowerCase();
}

function messageMediaSize(message) {
  const data = message && message._data ? message._data : {};
  const candidates = [data.size, data.fileSize];
  for (const candidate of candidates) {
    const parsed = Number(candidate);
    if (Number.isFinite(parsed) && parsed >= 0) {
      return parsed;
    }
  }
  return null;
}

function mediaExtension(mimeType, messageType) {
  const raw = String(mimeType || "").split(";", 1)[0];
  const subtype = raw.includes("/") ? raw.split("/")[1] : "";
  const cleaned = subtype.replace(/[^A-Za-z0-9]/g, "");
  if (cleaned) {
    return cleaned === "plain" ? "txt" : cleaned;
  }
  if (messageType === "ptt" || messageType === "audio") {
    return "ogg";
  }
  return "bin";
}

function decodedBase64Size(value) {
  const data = String(value || "");
  const padding = data.endsWith("==") ? 2 : data.endsWith("=") ? 1 : 0;
  return Math.floor((data.length * 3) / 4) - padding;
}

function normalizeIds(values) {
  return values.map(widString).filter(Boolean);
}

function safeVersion(value) {
  const version = String(value || "unknown").replace(/[^0-9A-Za-z._-]/g, "");
  return version.slice(0, 64) || "unknown";
}

function cacheSentMessage(message) {
  normalizeMessage(message);
  const id = serializedId(message && message.id);
  if (!id) {
    return null;
  }
  sentMessageIds.add(id);
  sentMessages.set(id, message);
  if (sentMessages.size > MAX_CACHED_SENT_MESSAGES) {
    const oldest = sentMessages.keys().next().value;
    sentMessages.delete(oldest);
    sentMessageIds.delete(oldest);
    ackTracker.forget(oldest);
  }
  return id;
}

function rememberSentMessage(message) {
  const id = cacheSentMessage(message);
  if (!id) {
    return null;
  }
  ackTracker.register(id, message.ack);
  return id;
}

function recordMessageAck(message, ack) {
  normalizeMessage(message);
  ackTracker.record(serializedId(message && message.id), ack);
}

function isBridgeSentMessage(message) {
  const id = serializedId(message.id);
  if (id && sentMessageIds.has(id)) {
    return true;
  }
  return sentBodies.has(message.body || "");
}

async function withSentBodyReservation(body, operation) {
  const release = sentBodies.reserve(body);
  try {
    return await operation();
  } finally {
    release();
  }
}

async function withBridgeMediaSend(operation) {
  bridgeMediaSends += 1;
  try {
    return await operation();
  } finally {
    bridgeMediaSends -= 1;
  }
}

function readJson(req) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    req.on("data", (chunk) => chunks.push(chunk));
    req.on("end", () => {
      if (chunks.length === 0) {
        resolve({});
        return;
      }
      try {
        resolve(JSON.parse(Buffer.concat(chunks).toString("utf8")));
      } catch (error) {
        reject(error);
      }
    });
    req.on("error", reject);
  });
}

function sendJson(res, code, body) {
  const data = Buffer.from(JSON.stringify(body));
  res.writeHead(code, {
    "content-type": "application/json",
    "content-length": String(data.length),
  });
  res.end(data);
}

function isAuthorized(req) {
  return req.headers.authorization === `Bearer ${bridgeToken}`;
}

function containedMediaPath(value) {
  if (typeof value !== "string" || !value) {
    return null;
  }
  const resolved = path.resolve(value);
  const root = path.resolve(mediaDir);
  if (resolved === root || !resolved.startsWith(root + path.sep)) {
    return null;
  }
  return resolved;
}

async function handle(req, res) {
  try {
    if (!isAuthorized(req)) {
      sendJson(res, 401, { success: false, error: "unauthorized" });
      return;
    }

    if (req.method === "GET" && req.url === "/health") {
      sendJson(res, 200, { status, botId });
      return;
    }

    if (req.method === "GET" && req.url === "/messages") {
      sendJson(res, 200, queue.splice(0, queue.length));
      return;
    }

    if (req.method === "POST" && status !== "connected") {
      sendJson(res, 503, { success: false, error: "WhatsApp bridge is not connected" });
      return;
    }

    if (req.method === "POST" && req.url === "/send") {
      const body = await readJson(req);
      const chatId = body.chat_id || body.chatId;
      const text = body.text || body.message || "";
      if (!chatId || !text) {
        sendJson(res, 400, { success: false, error: "chat_id and text required" });
        return;
      }
      const messageId = await withSentBodyReservation(text, async () => {
        const sent = await client.sendMessage(chatId, text, {
          quotedMessageId: body.replyTo || body.reply_to || undefined,
        });
        return rememberSentMessage(sent);
      });
      if (!messageId) {
        sendJson(res, 502, { success: false, error: "WhatsApp send returned no message id" });
        return;
      }
      sendJson(res, 200, {
        success: true,
        message_id: messageId,
        messageId,
      });
      return;
    }

    if (req.method === "POST" && req.url === "/send-media") {
      const body = await readJson(req);
      const chatId = body.chat_id || body.chatId;
      const filePath = body.path || body.filePath;
      if (!chatId || !filePath) {
        sendJson(res, 400, { success: false, error: "chat_id and path required" });
        return;
      }
      const safePath = containedMediaPath(filePath);
      if (!safePath) {
        sendJson(res, 400, { success: false, error: "media path is not allowed" });
        return;
      }
      const media = MessageMedia.fromFilePath(safePath);
      if (body.fileName || body.file_name) {
        media.filename = body.fileName || body.file_name;
      }
      const caption = body.caption || undefined;
      const messageId = await withBridgeMediaSend(() =>
        withSentBodyReservation(caption, async () => {
          const sent = await client.sendMessage(chatId, media, {
            caption,
            sendMediaAsDocument: body.mediaType === "document",
          });
          return rememberSentMessage(sent);
        }),
      );
      if (!messageId) {
        sendJson(res, 502, { success: false, error: "WhatsApp send returned no message id" });
        return;
      }
      sendJson(res, 200, {
        success: true,
        message_id: messageId,
        messageId,
      });
      return;
    }

    if (req.method === "POST" && req.url === "/typing") {
      const body = await readJson(req);
      const chatId = body.chat_id || body.chatId;
      if (!chatId) {
        sendJson(res, 400, { success: false, error: "chat_id required" });
        return;
      }
      const chat = await client.getChatById(chatId);
      await chat.sendStateTyping();
      sendJson(res, 200, { success: true, ok: true });
      return;
    }

    if (req.method === "POST" && req.url === "/edit") {
      const body = await readJson(req);
      const messageId = body.message_id || body.messageId;
      const content = body.content || body.message || "";
      if (!messageId || !content) {
        sendJson(res, 400, { success: false, error: "message_id and content required" });
        return;
      }
      const message = sentMessages.get(messageId) || (await client.getMessageById(messageId));
      if (!message) {
        sendJson(res, 200, { success: false, error: "message not found" });
        return;
      }
      normalizeMessage(message);
      const edited = await withSentBodyReservation(content, () => message.edit(content));
      if (!edited) {
        sendJson(res, 200, { success: false, error: "message could not be edited" });
        return;
      }
      const editedId = cacheSentMessage(edited) || messageId;
      sendJson(res, 200, {
        success: true,
        message_id: editedId,
        messageId: editedId,
      });
      return;
    }

    sendJson(res, 404, { success: false, error: "not found" });
  } catch (error) {
    sendJson(res, 500, { success: false, error: error.message || String(error) });
  }
}

const server = http.createServer((req, res) => {
  void handle(req, res);
});

server.listen(port, host, () => {
  console.log(`WhatsApp bridge listening on http://${host}:${port}`);
});

client.initialize();

process.on("SIGTERM", async () => {
  server.close();
  try {
    await client.destroy();
  } catch (_error) {
    // The process is already exiting.
  }
  process.exit(0);
});
