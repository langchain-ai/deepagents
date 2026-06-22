"use strict";

const fs = require("fs");
const http = require("http");
const path = require("path");
const { Client, LocalAuth, MessageMedia } = require("whatsapp-web.js");
const qrcode = require("qrcode-terminal");

const host = process.env.WHATSAPP_BRIDGE_HOST || "127.0.0.1";
const port = Number(process.env.WHATSAPP_BRIDGE_PORT || "3000");
const sessionDir = path.resolve(process.env.WHATSAPP_SESSION_DIR || path.join(process.cwd(), ".whatsapp"));
const mediaDir = path.resolve(process.env.WHATSAPP_MEDIA_DIR || path.join(sessionDir, "..", "media"));
const botHeader = process.env.WHATSAPP_BOT_HEADER || "deepagents bot";
const bridgeToken = process.env.WHATSAPP_BRIDGE_TOKEN || "";
const maxMediaBytes = Number(process.env.WHATSAPP_MAX_MEDIA_BYTES) || (64 * 1024 * 1024);
const webVersionCacheUrl =
  process.env.WHATSAPP_WEB_VERSION_CACHE_URL ||
  "https://raw.githubusercontent.com/wppconnect-team/wa-version/main/html/2.3000.1026029003.html";

const MAX_CACHED_SENT_MESSAGES = 200;
const SENT_BODY_TTL_MS = 5 * 60 * 1000;

let status = "disconnected";
let botId = null;
const queue = [];
const sentMessageIds = new Set();
const sentMessages = new Map();
const recentSentBodies = new Map();

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
  console.log(`Using Chrome at: ${chromePath}`);
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

const client = new Client({
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
  status = "connected";
  botId = client.info && client.info.wid ? client.info.wid._serialized : null;
  console.log(`WhatsApp connected as ${botId || "unknown"}`);
});

client.on("disconnected", (reason) => {
  status = "disconnected";
  console.log(`WhatsApp disconnected: ${reason || "unknown reason"}`);
});

client.on("auth_failure", (message) => {
  status = "disconnected";
  console.error(`WhatsApp auth failure: ${message || "unknown error"}`);
});

client.on("message_create", (message) => {
  void enqueueMessage(message);
});

async function enqueueMessage(message) {
  if (message.from === "status@broadcast") {
    return;
  }
  const fromSelf = isSelfMessage(message);
  if (fromSelf && isBridgeSentMessage(message)) {
    return;
  }

  const chat = await safeGetChat(message);
  const contact = await safeGetContact(message);
  const messageId = serializedId(message.id);
  const chatId = serializedId(chat && chat.id) || (fromSelf ? message.to : message.from);
  if (!messageId || !chatId) {
    console.error("Skipping WhatsApp message without a message id or chat id");
    return;
  }

  const media = await downloadMessageMedia(message);
  const mediaType = classifyMedia(message, media);
  if (message.hasMedia && media.length === 0) {
    console.log(
      `[bridge] Message ${messageId} reported media but no attachment was downloaded; type=${message.type || "unknown"} mediaType=${mediaType}`,
    );
  }
  const senderId = message.author || (fromSelf && botId ? botId : message.from);
  const isGroup =
    chat && typeof chat.isGroup === "boolean"
      ? chat.isGroup
      : typeof chatId === "string" && chatId.endsWith("@g.us");

  const entry = {
    text: message.body || "",
    body: message.body || "",
    message_type: message.type || "chat",
    messageType: message.type || "chat",
    media_type: mediaType,
    mediaType,
    chat_id: chatId,
    chatId,
    chat_id_from: message.from,
    chatIdFrom: message.from,
    chat_name: (chat && chat.name) || chatId,
    chatName: (chat && chat.name) || chatId,
    chat_type: isGroup ? "group" : "direct",
    chatType: isGroup ? "group" : "direct",
    isGroup,
    user_id: senderId || null,
    senderId: senderId || null,
    user_name:
      (contact && (contact.pushname || contact.name || contact.shortName)) || senderId || null,
    senderName:
      (contact && (contact.pushname || contact.name || contact.shortName)) || senderId || null,
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
    mentionedIds: normalizeIds(message.mentionedIds || []),
    botIds: botId ? [botId] : [],
    quotedParticipant: await quotedParticipant(message),
    raw_message: {
      from: message.from,
      to: message.to,
      author: message.author || null,
      fromMe: Boolean(message.fromMe),
      idFromMe: Boolean(message.id && message.id.fromMe),
      timestamp: message.timestamp || null,
    },
  };

  console.log(`[bridge] Queued message ${messageId} for ${chatId}`);
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
  } catch (error) {
    console.log(`[bridge] getChat failed (non-fatal): ${error.message || error}`);
    return null;
  }
}

async function safeGetContact(message) {
  try {
    return await message.getContact();
  } catch (error) {
    console.log(`[bridge] getContact failed (non-fatal): ${error.message || error}`);
    return null;
  }
}

async function quotedParticipant(message) {
  if (!message.hasQuotedMsg) {
    return null;
  }
  try {
    const quoted = await message.getQuotedMessage();
    return quoted.author || quoted.from || null;
  } catch (_error) {
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
        `[bridge] Skipping oversized media ${messageId}: ${expectedSize} bytes exceeds ${maxMediaBytes}`,
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
        `[bridge] Skipping oversized media ${messageId}: ${size} bytes exceeds ${maxMediaBytes}`,
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

function serializedId(value) {
  return value && value._serialized ? value._serialized : null;
}

function normalizeIds(values) {
  return values.map((value) => (typeof value === "object" ? value._serialized : value)).filter(Boolean);
}

function isSelfMessage(message) {
  if (message.fromMe === true || (message.id && message.id.fromMe === true)) {
    return true;
  }
  const id = serializedId(message.id);
  return typeof id === "string" && id.startsWith("true_");
}

function rememberSentMessage(message, body) {
  const id = serializedId(message && message.id);
  if (!id) {
    return;
  }
  sentMessageIds.add(id);
  sentMessages.set(id, message);
  rememberSentBody(body);
  if (sentMessages.size > MAX_CACHED_SENT_MESSAGES) {
    const oldest = sentMessages.keys().next().value;
    sentMessages.delete(oldest);
  }
}

function rememberSentBody(body) {
  if (!body) {
    return;
  }
  const key = String(body);
  recentSentBodies.set(key, (recentSentBodies.get(key) || 0) + 1);
  setTimeout(() => {
    const count = recentSentBodies.get(key) || 0;
    if (count <= 1) {
      recentSentBodies.delete(key);
    } else {
      recentSentBodies.set(key, count - 1);
    }
  }, SENT_BODY_TTL_MS).unref();
}

function isBridgeSentMessage(message) {
  const id = serializedId(message.id);
  if (id && sentMessageIds.has(id)) {
    return true;
  }
  const body = message.body || "";
  if (body && recentSentBodies.has(body)) {
    return true;
  }
  return Boolean(body && body.startsWith(`*${botHeader}*`));
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

    if (req.method === "POST" && req.url === "/send") {
      const body = await readJson(req);
      const chatId = body.chat_id || body.chatId;
      const text = body.text || body.message || "";
      if (!chatId || !text) {
        sendJson(res, 400, { success: false, error: "chat_id and text required" });
        return;
      }
      rememberSentBody(text);
      const sent = await client.sendMessage(chatId, text, {
        quotedMessageId: body.replyTo || body.reply_to || undefined,
      });
      rememberSentMessage(sent, text);
      const messageId = serializedId(sent.id);
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
      if (caption) {
        rememberSentBody(caption);
      }
      const sent = await client.sendMessage(chatId, media, {
        caption,
        sendMediaAsDocument: body.mediaType === "document",
      });
      rememberSentMessage(sent, caption || "");
      const messageId = serializedId(sent.id);
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
      rememberSentBody(content);
      const edited = await message.edit(content);
      rememberSentMessage(edited, content);
      const editedId = serializedId(edited.id) || messageId;
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
