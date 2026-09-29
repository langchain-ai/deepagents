"use strict";

const assert = require("node:assert/strict");
const test = require("node:test");
const vm = require("node:vm");
const { downloadMedia } = require("../../../deepagents_talon/channels/whatsapp_bridge/media");

const messageId = "false_test@c.us_VOICE";
const inbound = { hasMedia: true, id: { _serialized: messageId } };
const audio = Buffer.from("test audio");

function voiceMessage(stage = "RESOLVED") {
  return {
    type: "ptt",
    mimetype: "audio/ogg; codecs=opus",
    mediaData: { mediaStage: stage },
    directPath: "/test-audio",
    encFilehash: "test-encrypted-hash",
    filehash: "test-hash",
    mediaKey: "test-key",
    mediaKeyTimestamp: 1,
    filename: "voice.ogg",
    size: audio.length,
  };
}

function browserPage(message, { cached = true, download, lookup } = {}) {
  const window = {
    require(name) {
      if (name === "WAWebCollections") {
        return {
          Msg: {
            get: (id) => cached && id === messageId ? message : undefined,
            getMessagesById: lookup || (async (ids) => ({
              messages: ids[0] === messageId && message ? [message] : [],
            })),
          },
        };
      }
      assert.equal(name, "WAWebDownloadManager");
      return { downloadManager: { downloadAndMaybeDecrypt: download || decryptAudio } };
    },
    WWebJS: {
      arrayBufferToBase64Async: async (data) => Buffer.from(data).toString("base64"),
    },
  };
  return {
    evaluate: async (callback, id) => vm.runInNewContext(
      `(${callback.toString()})(messageId)`,
      { window, AbortController, messageId: id },
    ),
  };
}

async function decryptAudio(options) {
  if (options.mimetype !== "audio/ogg; codecs=opus") {
    throw new Error("Unexpected mimetype application/octet-stream for media type ptt");
  }
  return audio;
}

for (const cached of [true, false]) {
  test(`downloads voice audio with its MIME type from ${cached ? "cache" : "message lookup"}`, async () => {
    await assert.rejects(decryptAudio({ type: "ptt" }), /Unexpected mimetype/);
    const media = await downloadMedia(browserPage(voiceMessage(), { cached }), inbound);
    assert.equal(media.data, audio.toString("base64"));
    assert.equal(media.mimetype, "audio/ogg; codecs=opus");
    assert.equal(media.filename, "voice.ogg");
    assert.equal(media.filesize, audio.length);
  });
}

test("does not evaluate the page for messages without media", async () => {
  const page = { evaluate: () => assert.fail("unexpected page evaluation") };
  assert.equal(await downloadMedia(page, { hasMedia: false }), undefined);
});

for (const [label, message] of [
  ["missing message", undefined],
  ["missing media data", { type: "ptt" }],
  ["reuploading media", voiceMessage("REUPLOADING")],
]) {
  test(`skips ${label}`, async () => {
    const page = browserPage(message, { download: () => assert.fail("unexpected download") });
    assert.equal(await downloadMedia(page, inbound), undefined);
  });
}

test("handles an empty lookup result", async () => {
  const page = browserPage(undefined, { lookup: async () => undefined });
  assert.equal(await downloadMedia(page, inbound), undefined);
});

test("resolves pending media before downloading", async () => {
  const message = voiceMessage("PENDING");
  message.downloadMedia = async () => {
    message.mediaData.mediaStage = "RESOLVED";
  };
  const media = await downloadMedia(browserPage(message), inbound);
  assert.equal(media.data, audio.toString("base64"));
});

for (const stage of ["FETCHING", "ERROR", "ERROR_RETRYABLE"]) {
  test(`skips media still in ${stage} after resolution`, async () => {
    const message = voiceMessage("PENDING");
    message.downloadMedia = async () => {
      message.mediaData.mediaStage = stage;
    };
    const page = browserPage(message, { download: () => assert.fail("unexpected download") });
    assert.equal(await downloadMedia(page, inbound), undefined);
  });
}

test("returns unavailable media on a 404", async () => {
  const download = async () => { throw Object.assign(new Error("missing"), { status: 404 }); };
  assert.equal(await downloadMedia(browserPage(voiceMessage(), { download }), inbound), undefined);
});

test("propagates download failures other than 404", async () => {
  const error = Object.assign(new Error("download failed"), { status: 500 });
  const download = async () => { throw error; };
  await assert.rejects(downloadMedia(browserPage(voiceMessage(), { download }), inbound), error);
});

test("propagates media resolution failures", async () => {
  const error = new Error("resolution failed");
  const message = voiceMessage("PENDING");
  message.downloadMedia = async () => { throw error; };
  await assert.rejects(downloadMedia(browserPage(message), inbound), error);
});
