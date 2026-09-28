"use strict";

async function downloadMedia(page, message) {
  if (!message.hasMedia) {
    return undefined;
  }
  const media = await page.evaluate(downloadBrowserMedia, message.id._serialized);
  return media || undefined;
}

async function downloadBrowserMedia(messageId) {
  const messages = window.require("WAWebCollections").Msg;
  const message =
    messages.get(messageId) ||
    (await messages.getMessagesById([messageId]))?.messages?.[0];
  if (!message || !message.mediaData || message.mediaData.mediaStage === "REUPLOADING") {
    return null;
  }
  if (message.mediaData.mediaStage !== "RESOLVED") {
    await message.downloadMedia({ downloadEvenIfExpensive: true, rmrReason: 1 });
  }
  if (
    message.mediaData.mediaStage.includes("ERROR") ||
    message.mediaData.mediaStage === "FETCHING"
  ) {
    return undefined;
  }
  try {
    const downloadQpl = {
      addAnnotations() {
        return this;
      },
      addPoint() {
        return this;
      },
    };
    const decrypted = await window.require("WAWebDownloadManager")
      .downloadManager.downloadAndMaybeDecrypt({
        directPath: message.directPath,
        encFilehash: message.encFilehash,
        filehash: message.filehash,
        mediaKey: message.mediaKey,
        mediaKeyTimestamp: message.mediaKeyTimestamp,
        type: message.type,
        mimetype: message.mimetype,
        signal: new AbortController().signal,
        downloadQpl,
      });
    return {
      data: await window.WWebJS.arrayBufferToBase64Async(decrypted),
      mimetype: message.mimetype,
      filename: message.filename,
      filesize: message.size,
    };
  } catch (error) {
    if (error.status === 404) {
      return undefined;
    }
    throw error;
  }
}

module.exports = { downloadMedia };
