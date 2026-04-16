const { Client, LocalAuth, MessageMedia } = require("whatsapp-web.js");
const qrcode = require("qrcode-terminal");
const express = require("express");
const path = require("path");
const fs = require("fs");

// --- CLI args ---
const args = process.argv.slice(2);
function getArg(name, defaultVal) {
  const idx = args.indexOf(`--${name}`);
  return idx !== -1 && args[idx + 1] ? args[idx + 1] : defaultVal;
}
const PORT = parseInt(getArg("port", "3000"), 10);
const SESSION_DIR = path.resolve(getArg("session", "./session"));
const MEDIA_DIR = path.resolve(getArg("media-dir", "./media"));

// Ensure directories exist
fs.mkdirSync(SESSION_DIR, { recursive: true });
fs.mkdirSync(MEDIA_DIR, { recursive: true });

// --- State ---
let clientStatus = "disconnected"; // "disconnected" | "qr_pending" | "connected"
let botId = null;
const messageQueue = [];

// --- WhatsApp client ---
const client = new Client({
  authStrategy: new LocalAuth({ dataPath: SESSION_DIR }),
  puppeteer: {
    headless: true,
    args: ["--no-sandbox", "--disable-setuid-sandbox"],
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

client.on("message", async (msg) => {
  // Skip status broadcasts
  if (msg.from === "status@broadcast") return;
  // Skip messages sent by the bot itself
  if (msg.fromMe) return;

  const chat = await msg.getChat();
  const contact = await msg.getContact();

  const entry = {
    messageId: msg.id._serialized,
    chatId: chat.id._serialized,
    chatName: chat.name || chat.id._serialized,
    senderId: msg.author || msg.from,
    senderName: contact.pushname || contact.name || msg.from,
    body: msg.body || "",
    isGroup: chat.isGroup,
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
    // whatsapp-web.js doesn't have a direct edit API on the client;
    // we need the original message object. For prototype, return not-supported.
    res.status(501).json({ error: "edit not supported by whatsapp-web.js" });
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
