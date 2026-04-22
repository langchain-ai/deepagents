const { Client, LocalAuth, MessageMedia } = require("whatsapp-web.js");
const qrcode = require("qrcode-terminal");
const express = require("express");
const path = require("path");
const fs = require("fs");

// whatsapp-web.js can throw "data passed to getter must include an id property"
// from its Store module when a message event fires with partial state (common
// after a WhatsApp Web update vs. the pinned library version). Those become
// unhandled rejections from our async handlers — swallow here so one bad
// event never crashes the bridge.
process.on("unhandledRejection", (reason) => {
  console.error("Unhandled rejection:", reason && reason.message ? reason.message : reason);
});

// --- CLI args ---
const args = process.argv.slice(2);
function getArg(name, defaultVal) {
  const idx = args.indexOf(`--${name}`);
  return idx !== -1 && args[idx + 1] ? args[idx + 1] : defaultVal;
}
function hasFlag(name) {
  return args.includes(`--${name}`);
}
const PORT = parseInt(getArg("port", "3000"), 10);
const SESSION_DIR = path.resolve(getArg("session", "./session"));
const MEDIA_DIR = path.resolve(getArg("media-dir", "./media"));
const SELF_ONLY = hasFlag("self-only");

const CHROME_PROFILE_DIR = path.join(SESSION_DIR, "chromium-profile");

// Ensure directories exist
fs.mkdirSync(SESSION_DIR, { recursive: true });
fs.mkdirSync(MEDIA_DIR, { recursive: true });
fs.mkdirSync(CHROME_PROFILE_DIR, { recursive: true });

// Remove stale Chromium lock files left from previous runs/containers.
// LocalAuth keeps its profile under SESSION_DIR/session/, so we recurse
// rather than targeting a single known path.
function cleanStaleLocks(dir) {
  let entries;
  try { entries = fs.readdirSync(dir, { withFileTypes: true }); } catch (_) { return; }
  for (const entry of entries) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      cleanStaleLocks(full);
    } else if (/^Singleton(Lock|Socket|Cookie)$/.test(entry.name)) {
      try { fs.unlinkSync(full); } catch (_) {}
    }
  }
}
cleanStaleLocks(SESSION_DIR);

// --- State ---
let clientStatus = "disconnected"; // "disconnected" | "qr_pending" | "connected"
let botId = null;
const messageQueue = [];
const sentMessageIds = new Set(); // track IDs we sent to avoid reply loops
const sentMessages = new Map(); // messageId -> Message object (for editing)
const MAX_CACHED_MESSAGES = 200;

// --- Detect system Chrome/Chromium ---
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
            process.env.PROGRAMFILES + "\\Google\\Chrome\\Application\\chrome.exe",
            process.env["PROGRAMFILES(X86)"] + "\\Google\\Chrome\\Application\\chrome.exe",
          ]
        : [
            "/usr/bin/google-chrome",
            "/usr/bin/google-chrome-stable",
            "/usr/bin/chromium",
            "/usr/bin/chromium-browser",
            "/snap/bin/chromium",
          ];
  for (const c of candidates) {
    if (c && fs.existsSync(c)) return c;
  }
  return undefined; // fall back to puppeteer's bundled browser if available
}

const chromePath = process.env.CHROME_PATH || findChrome();
if (chromePath) {
  console.log(`Using Chrome at: ${chromePath}`);
} else {
  console.log("No system Chrome found; using puppeteer bundled browser (if available)");
}

// --- WhatsApp client ---
const puppeteerOpts = {
  headless: true,
  args: [
    "--no-sandbox",
    "--disable-setuid-sandbox",
    "--disable-gpu",
    // Persistent Chromium profile inside the session dir (stale locks cleaned above)
    `--user-data-dir=${CHROME_PROFILE_DIR}`,
  ],
};
if (chromePath) {
  puppeteerOpts.executablePath = chromePath;
}

const client = new Client({
  authStrategy: new LocalAuth({ dataPath: SESSION_DIR }),
  puppeteer: puppeteerOpts,
  // whatsapp-web.js is tightly coupled to the WhatsApp Web HTML/JS shipped by
  // the server; when the server ships a newer build than the library knows
  // about, Store getters start throwing "must include an id property". Loading
  // a pinned HTML from wa-version bypasses the broken server build.
  // See https://github.com/pedroslopez/whatsapp-web.js/issues/3120
  webVersionCache: {
    type: "remote",
    remotePath:
      "https://raw.githubusercontent.com/wppconnect-team/wa-version/main/html/2.3000.1026029003.html",
  },
});

client.on("qr", (qr) => {
  clientStatus = "qr_pending";
  console.log("Scan this QR code to log in:");
  qrcode.generate(qr, { small: true });
});

client.on("ready", () => {
  clientStatus = "connected";
  botId = client.info.wid._serialized;
  console.log(`WhatsApp connected as ${botId}`);
});

client.on("disconnected", (reason) => {
  clientStatus = "disconnected";
  console.log(`WhatsApp disconnected: ${reason}`);
});

client.on("auth_failure", (msg) => {
  clientStatus = "disconnected";
  console.error(`Auth failure: ${msg}`);
});

// Use message_create (fires for both sent and received messages) so that
// self-chat messages (fromMe=true) are captured.  The "message" event only
// fires for received messages, which misses self-chat in self-only mode.
client.on("message_create", async (msg) => {
  // Skip status broadcasts
  if (msg.from === "status@broadcast") return;

  if (SELF_ONLY) {
    // Self-only mode: only process messages from our own number
    if (!msg.fromMe) return;
    // Skip bot replies by checking for the bot header prefix
    if ((msg.body || "").startsWith("*deepagents bot*")) return;
  } else {
    // Normal mode: skip all self-sent messages
    if (msg.fromMe) return;
  }

  // Try the rich getters, but fall back to msg fields when WhatsApp Web's
  // Store is in a state whatsapp-web.js can't index into (the "must include
  // an id property" / "_serialized of undefined" class of errors). Chat id
  // in group messages is msg.from; in 1:1 it's also msg.from (from our side
  // it's msg.to, but we've already skipped fromMe above unless SELF_ONLY).
  let chat = null;
  let contact = null;
  try { chat = await msg.getChat(); } catch (e) {
    console.error(`getChat failed for ${msg.from}: ${e.message}`);
  }
  try { contact = await msg.getContact(); } catch (e) {
    console.error(`getContact failed for ${msg.from}: ${e.message}`);
  }

  const chatIdSerialized =
    (chat && chat.id && chat.id._serialized) || msg.from;
  const isGroup =
    (chat && typeof chat.isGroup === "boolean"
      ? chat.isGroup
      : typeof chatIdSerialized === "string" && chatIdSerialized.endsWith("@g.us"));

  const entry = {
    messageId: (msg.id && msg.id._serialized) || null,
    chatId: chatIdSerialized,
    chatName: (chat && chat.name) || chatIdSerialized,
    senderId: msg.author || msg.from,
    senderName:
      (contact && (contact.pushname || contact.name)) || msg.from,
    body: msg.body || "",
    isGroup,
    hasMedia: msg.hasMedia,
    mediaType: msg.type !== "chat" ? msg.type : null,
    mediaUrls: [],
    mentionedIds: (msg.mentionedIds || []).map((id) =>
      typeof id === "object" ? id._serialized : id
    ),
    botIds: botId ? [botId] : [],
    quotedParticipant: null,
    timestamp: msg.timestamp,
  };

  // Drop if we couldn't resolve the bare minimum to route the message.
  if (!entry.messageId || !entry.chatId) {
    console.error("Skipping message with no resolvable id/chat");
    return;
  }

  // Handle quoted message
  if (msg.hasQuotedMsg) {
    try {
      const quoted = await msg.getQuotedMessage();
      entry.quotedParticipant = quoted.author || quoted.from;
    } catch (_) {}
  }

  // Download media if present
  if (msg.hasMedia) {
    try {
      const media = await msg.downloadMedia();
      if (media) {
        const ext = media.mimetype ? media.mimetype.split("/")[1] : "bin";
        const filename = `${Date.now()}_${msg.id._serialized.replace(/[^a-zA-Z0-9]/g, "_")}.${ext}`;
        const filepath = path.join(MEDIA_DIR, filename);
        fs.writeFileSync(filepath, Buffer.from(media.data, "base64"));
        entry.mediaUrls.push(filepath);
      }
    } catch (e) {
      console.error("Media download failed:", e.message);
    }
  }

  messageQueue.push(entry);
});

// --- HTTP server ---
const app = express();
app.use(express.json({ limit: "50mb" }));

app.get("/health", (_req, res) => {
  res.json({ status: clientStatus, botId });
});

app.get("/messages", (_req, res) => {
  const messages = messageQueue.splice(0, messageQueue.length);
  res.json(messages);
});

app.post("/send", async (req, res) => {
  const { chatId, message, replyTo } = req.body;
  if (!chatId || !message) {
    return res.status(400).json({ error: "chatId and message required" });
  }
  try {
    const options = {};
    if (replyTo) options.quotedMessageId = replyTo;
    const sent = await client.sendMessage(chatId, message, options);
    sentMessageIds.add(sent.id._serialized);
    sentMessages.set(sent.id._serialized, sent);
    // Evict oldest cached message to prevent memory leaks
    if (sentMessages.size > MAX_CACHED_MESSAGES) {
      const oldest = sentMessages.keys().next().value;
      sentMessages.delete(oldest);
    }
    res.json({ messageId: sent.id._serialized });
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

app.post("/send-media", async (req, res) => {
  const { chatId, filePath, mediaType, caption, fileName } = req.body;
  if (!chatId || !filePath) {
    return res.status(400).json({ error: "chatId and filePath required" });
  }
  try {
    const media = MessageMedia.fromFilePath(filePath);
    if (fileName) media.filename = fileName;
    const options = {};
    if (caption) options.caption = caption;
    if (mediaType === "document") options.sendMediaAsDocument = true;
    const sent = await client.sendMessage(chatId, media, options);
    sentMessageIds.add(sent.id._serialized);
    res.json({ messageId: sent.id._serialized });
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

app.post("/typing", async (req, res) => {
  const { chatId } = req.body;
  if (!chatId) return res.status(400).json({ error: "chatId required" });
  try {
    const chat = await client.getChatById(chatId);
    await chat.sendStateTyping();
    res.json({ ok: true });
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

app.post("/edit", async (req, res) => {
  const { chatId, messageId, message } = req.body;
  if (!chatId || !messageId || !message) {
    return res.status(400).json({ error: "chatId, messageId, and message required" });
  }
  try {
    const msg = sentMessages.get(messageId);
    if (!msg) {
      return res.status(404).json({ error: "Message not found in sent cache" });
    }
    await msg.edit(message);
    res.json({ ok: true, messageId });
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

app.get("/chat/:id", async (req, res) => {
  try {
    const chat = await client.getChatById(req.params.id);
    const participants = chat.participants
      ? chat.participants.map((p) => ({
          id: p.id._serialized,
          isAdmin: p.isAdmin || false,
        }))
      : [];
    res.json({
      name: chat.name || req.params.id,
      isGroup: chat.isGroup,
      participants,
    });
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

// --- Start ---
app.listen(PORT, "127.0.0.1", () => {
  console.log(`Bridge HTTP server listening on 127.0.0.1:${PORT}`);
});

client.initialize();
