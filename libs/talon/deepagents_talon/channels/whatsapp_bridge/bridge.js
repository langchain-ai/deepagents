"use strict";

const http = require("http");
const path = require("path");
const { Client, LocalAuth, MessageMedia } = require("whatsapp-web.js");
const qrcode = require("qrcode-terminal");

const host = process.env.WHATSAPP_BRIDGE_HOST || "127.0.0.1";
const port = Number(process.env.WHATSAPP_BRIDGE_PORT || "3000");
const sessionDir = process.env.WHATSAPP_SESSION_DIR || path.join(process.cwd(), ".whatsapp");

let status = "disconnected";
let botId = null;
const queue = [];

const client = new Client({
  authStrategy: new LocalAuth({
    dataPath: sessionDir,
  }),
  puppeteer: {
    args: ["--no-sandbox", "--disable-setuid-sandbox"],
  },
});

client.on("qr", (qr) => {
  status = "qr_pending";
  qrcode.generate(qr, { small: true });
});

client.on("ready", () => {
  status = "connected";
  botId = client.info && client.info.wid ? client.info.wid._serialized : null;
});

client.on("disconnected", () => {
  status = "disconnected";
});

client.on("auth_failure", () => {
  status = "disconnected";
});

client.on("message", async (message) => {
  const chat = await message.getChat();
  const contact = await message.getContact();
  queue.push({
    text: message.body || "",
    message_type: message.type || "text",
    chat_id: chat.id && chat.id._serialized ? chat.id._serialized : message.from,
    chat_name: chat.name || null,
    chat_type: chat.isGroup ? "group" : "direct",
    user_id: contact.id && contact.id._serialized ? contact.id._serialized : null,
    user_name: contact.pushname || contact.name || null,
    message_id: message.id && message.id._serialized ? message.id._serialized : null,
    media_urls: [],
    media_types: message.hasMedia ? [message.type] : [],
    from_self: Boolean(message.fromMe),
    raw_message: {
      from: message.from,
      to: message.to,
      author: message.author || null,
      timestamp: message.timestamp || null,
    },
  });
});

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

async function handle(req, res) {
  try {
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
      const sent = await client.sendMessage(body.chat_id, body.text || "", {
        quotedMessageId: body.replyTo || undefined,
      });
      sendJson(res, 200, {
        success: true,
        message_id: sent.id && sent.id._serialized ? sent.id._serialized : undefined,
      });
      return;
    }

    if (req.method === "POST" && req.url === "/send-media") {
      const body = await readJson(req);
      const media = MessageMedia.fromFilePath(body.path);
      const sent = await client.sendMessage(body.chat_id, media, {
        caption: body.caption || undefined,
      });
      sendJson(res, 200, {
        success: true,
        message_id: sent.id && sent.id._serialized ? sent.id._serialized : undefined,
      });
      return;
    }

    if (req.method === "POST" && req.url === "/edit") {
      const body = await readJson(req);
      const message = await client.getMessageById(body.message_id);
      if (!message) {
        sendJson(res, 200, { success: false, error: "message not found" });
        return;
      }
      const edited = await message.edit(body.content || "");
      sendJson(res, 200, {
        success: true,
        message_id: edited.id && edited.id._serialized ? edited.id._serialized : undefined,
      });
      return;
    }

    sendJson(res, 404, { success: false, error: "not found" });
  } catch (error) {
    sendJson(res, 500, { success: false, error: error.message || String(error) });
  }
}

const server = http.createServer((req, res) => {
  handle(req, res);
});

server.listen(port, host, () => {
  console.log(`WhatsApp bridge listening on http://${host}:${port}`);
});

client.initialize();

process.on("SIGTERM", async () => {
  server.close();
  await client.destroy();
  process.exit(0);
});
