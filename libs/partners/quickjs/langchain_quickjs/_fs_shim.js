// Installs `fs`, `require`, and the modules behind `import()` before every eval,
// replacing a restored snapshot's copy. Its host functions are frozen globals, so
// code cannot replace or shadow what `fs` calls.
((settle, fsSync, fsAsync, callAsync) => {
const call = (op, args) => settle(fsSync(op, args));
const acall = (op, args) => callAsync(() => fsAsync(op, args));

// Node returns a Buffer when no encoding is given, and its toString() decodes
// UTF-8, which code like `JSON.parse(fs.readFileSync(p))` relies on.
const bufferLike = (bytes) => {
  if (!(bytes instanceof Uint8Array)) return bytes;
  const toString = () => {
    let s = "";
    for (let i = 0; i < bytes.length; i += 8192) {
      s += String.fromCharCode.apply(null, bytes.subarray(i, i + 8192));
    }
    try {
      return decodeURIComponent(escape(s));
    } catch {
      return s;
    }
  };
  return Object.defineProperty(bytes, "toString", { value: toString, writable: true, configurable: true });
};

const normalize = (p) => {
  const abs = p.startsWith("/");
  const out = [];
  for (const s of p.split("/")) {
    if (!s || s === ".") continue;
    if (s !== "..") out.push(s);
    else if (out.length && out[out.length - 1] !== "..") out.pop();
    else if (!abs) out.push("..");
  }
  return (abs ? "/" : "") + out.join("/") || ".";
};
const resolve = (...args) => {
  const parts = args.filter(Boolean);
  return normalize("/" + parts.slice(Math.max(0, parts.findLastIndex((s) => s.startsWith("/")))).join("/"));
};
const basename = (p, ext) => {
  const b = normalize(p).split("/").pop();
  return ext && b !== ext && b.endsWith(ext) ? b.slice(0, -ext.length) : b;
};
const dirname = (p) => {
  const n = normalize(p);
  const i = n.lastIndexOf("/");
  return i < 0 ? "." : i === 0 ? "/" : n.slice(0, i);
};
const path = {
  sep: "/",
  isAbsolute: (p) => p.startsWith("/"),
  normalize,
  join: (...parts) => normalize(parts.filter((s) => s !== "").join("/") || "."),
  resolve,
  dirname,
  basename,
  extname: (p) => {
    const b = basename(p);
    const i = b.lastIndexOf(".");
    return i <= 0 ? "" : b.slice(i);
  },
};
path.posix = path;

const stringArg = (value, name) => {
  if (typeof value === "string") return value;
  const err = new TypeError(`The "${name}" argument must be of type string`);
  err.code = "ERR_INVALID_ARG_TYPE";
  throw err;
};
const pathArg = (p, name = "path") => resolve(stringArg(p, name));
const readArgs = (p, opts) => {
  const enc = typeof opts === "string" ? opts : opts?.encoding;
  return { path: pathArg(p), encoding: enc == null ? null : String(enc).toLowerCase() };
};
// Size and mtime come from the parent's listing, so they are best-effort.
const stats = (e) => {
  const mtime = e.mtime ? new Date(e.mtime) : undefined;
  return {
    size: e.size ?? undefined, mtime, mtimeMs: mtime?.getTime(),
    isFile: () => !e.isDir, isDirectory: () => e.isDir, isSymbolicLink: () => false,
  };
};
const ROOT = { isDir: true };
const dirents = (entries, dir, opts) => opts?.withFileTypes
  ? entries.map((e) => ({
    name: e.name, parentPath: dir, path: dir,
    isFile: () => !e.isDir, isDirectory: () => e.isDir, isSymbolicLink: () => false,
  }))
  : entries.map((e) => e.name);
const globArgs = (pattern, opts) => ({
  pattern: stringArg(pattern, "pattern"),
  path: opts?.cwd == null ? "/" : pathArg(opts.cwd, "cwd"),
});
const globPaths = (r) => {
  if (r.truncated) globalThis.console?.warn("fs.glob: the search stopped early; results are incomplete");
  return r.paths;
};

const promises = {
  readFile: async (p, opts) => bufferLike(await acall("readFile", readArgs(p, opts))),
  readdir: async (p, opts) => {
    const dir = pathArg(p);
    return dirents(await acall("readdir", { path: dir }), dir, opts);
  },
  glob: async function* (pattern, opts) {
    yield* globPaths(await acall("glob", globArgs(pattern, opts)));
  },
  stat: async (p) => {
    const abs = pathArg(p);
    return stats(abs === "/" ? ROOT : await acall("stat", { path: abs }));
  },
  access: async (p) => void (await promises.stat(p)),
};
promises.lstat = promises.stat;
const fs = {
  readFileSync: (p, opts) => bufferLike(call("readFile", readArgs(p, opts))),
  readdirSync: (p, opts) => {
    const dir = pathArg(p);
    return dirents(call("readdir", { path: dir }), dir, opts);
  },
  existsSync: (p) => {
    if (typeof p !== "string") return false;
    try {
      return Boolean(fs.statSync(p));
    } catch (e) {
      // Only a missing path is `false`; a failed listing is thrown.
      if (e?.code === "ENOENT") return false;
      throw e;
    }
  },
  globSync: (pattern, opts) => globPaths(call("glob", globArgs(pattern, opts))),
  statSync: (p) => {
    const abs = pathArg(p);
    return stats(abs === "/" ? ROOT : call("stat", { path: abs }));
  },
  accessSync: (p) => void fs.statSync(p),
  promises,
};
fs.lstatSync = fs.statSync;
const modules = { "node:fs": fs, "node:fs/promises": promises, "node:path": path };
// A top-level `function fs() {}` in code makes the global non-configurable;
// keep the code's own binding then rather than fail every later eval.
const define = (name, value) => {
  if (Object.getOwnPropertyDescriptor(globalThis, name)?.configurable === false) return;
  Object.defineProperty(globalThis, name, { value, writable: true, configurable: true, enumerable: false });
};
define("__lc_fs", modules);
define("fs", fs);
define("require", (id) => {
  const name = String(id).startsWith("node:") ? id : `node:${id}`;
  if (Object.prototype.hasOwnProperty.call(modules, name)) return modules[name];
  const err = new Error(`Cannot find module '${id}'`);
  err.code = "MODULE_NOT_FOUND";
  throw err;
});
})
