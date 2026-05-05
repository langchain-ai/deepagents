import { createRequire } from "node:module";
import { dirname, join } from "node:path";
import { pathToFileURL } from "node:url";

const decoder = new TextDecoder();
const encoder = new TextEncoder();

async function loadJustBash(specifier) {
  if (specifier.startsWith(".") || specifier.startsWith("/") || specifier.startsWith("file:")) {
    const url = specifier.startsWith("file:") ? specifier : pathToFileURL(specifier).href;
    return await import(url);
  }
  const require = createRequire(pathToFileURL(`${process.cwd()}/package.json`));
  const resolved = require.resolve(specifier);
  if (resolved.endsWith("/dist/bundle/index.cjs")) {
    return await import(pathToFileURL(join(dirname(resolved), "index.js")).href);
  }
  return await import(pathToFileURL(resolved).href);
}

function write(response) {
  process.stdout.write(`${JSON.stringify(response)}\n`);
}

function ok(value = {}) {
  write({ ok: true, ...value });
}

function fail(error) {
  write({ ok: false, error: error instanceof Error ? error.message : String(error) });
}

function normalizePath(path) {
  if (!path.startsWith("/")) {
    return `/${path}`;
  }
  return path;
}

function bytesToBase64(bytes) {
  return Buffer.from(bytes).toString("base64");
}

function base64ToBytes(value) {
  return new Uint8Array(Buffer.from(value, "base64"));
}

const specifier = process.env.JUST_BASH_PACKAGE ?? "just-bash";
const { Bash, InMemoryFs } = await loadJustBash(specifier);
const fs = new InMemoryFs();
const bash = new Bash({
  fs,
  javascript: process.env.JUST_BASH_JAVASCRIPT === "1",
});

async function readFile(path) {
  const content = await fs.readFileBuffer(normalizePath(path));
  return bytesToBase64(content);
}

async function writeFile(path, contentBase64) {
  await fs.writeFile(normalizePath(path), base64ToBytes(contentBase64));
}

async function statInfo(path) {
  const stat = await fs.stat(normalizePath(path));
  return {
    isDir: stat.isDirectory(),
    size: stat.size,
    mtime: stat.mtime instanceof Date ? stat.mtime.toISOString() : new Date(stat.mtime).toISOString(),
  };
}

async function walk(path, results) {
  const normalized = normalizePath(path);
  const stat = await fs.stat(normalized);
  if (!stat.isDirectory()) {
    const content = await readFile(normalized);
    results.push({ path: normalized, content, stat: await statInfo(normalized) });
    return;
  }

  for (const name of await fs.readdir(normalized)) {
    const child = normalized === "/" ? `/${name}` : `${normalized}/${name}`;
    await walk(child, results);
  }
}

async function listAll() {
  const files = [];
  await walk("/", files);
  return files;
}

async function handle(request) {
  switch (request.op) {
    case "execute": {
      const result = await bash.exec(request.command, {
        timeout: request.timeout === null || request.timeout === undefined ? undefined : request.timeout * 1000,
      });
      const stdout = result.stdout ?? "";
      const stderr = result.stderr ?? "";
      ok({
        output: stderr ? `${stdout}${stderr}` : stdout,
        exitCode: result.exitCode ?? 0,
      });
      return;
    }
    case "upload": {
      for (const file of request.files ?? []) {
        await writeFile(file.path, file.content);
      }
      ok({ responses: (request.files ?? []).map((file) => ({ path: file.path, error: null })) });
      return;
    }
    case "download": {
      const responses = [];
      for (const path of request.paths ?? []) {
        try {
          responses.push({ path, content: await readFile(path), error: null });
        } catch {
          responses.push({ path, content: null, error: "file_not_found" });
        }
      }
      ok({ responses });
      return;
    }
    case "files": {
      ok({ files: await listAll() });
      return;
    }
    default:
      fail(`unknown operation: ${request.op}`);
  }
}

let buffer = "";
process.stdin.setEncoding("utf8");
process.stdin.on("data", (chunk) => {
  buffer += chunk;
  for (;;) {
    const index = buffer.indexOf("\n");
    if (index === -1) {
      break;
    }
    const line = buffer.slice(0, index);
    buffer = buffer.slice(index + 1);
    if (!line.trim()) {
      continue;
    }
    void (async () => {
      try {
        await handle(JSON.parse(line));
      } catch (error) {
        fail(error);
      }
    })();
  }
});

process.on("SIGTERM", () => process.exit(0));
