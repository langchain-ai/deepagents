// Installs `fs`, `require`, and the modules behind `import()` over the
// host functions in `_fs.py`. Runs before every eval, so it replaces an older
// copy that a restored snapshot carried. `__ALIASES__` is replaced with
// `_MODULE_ALIASES` at import time.
(() => {
const unwrap = (r) => {
  const e = r !== null && typeof r === "object" ? r.__lc_fs_error : undefined;
  if (e === undefined) return r;
  const Ctor = e.code === "ERR_FS_FILE_TOO_LARGE" ? RangeError
    : e.code === "ERR_INVALID_ARG_VALUE" ? TypeError : Error;
  const err = new Ctor(e.message);
  for (const k of ["code", "syscall", "path"]) if (e[k] != null) err[k] = e[k];
  throw err;
};
const call = (op, args) => unwrap(globalThis.__lc_fs_sync(op, args));
const acall = async (op, args) => unwrap(await globalThis.__lc_fs_async(op, args));

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
  return (abs ? "/" : "") + out.join("/") || (abs ? "/" : ".");
};
const resolve = (...parts) => {
  let r = "";
  for (let i = parts.length - 1; i >= 0 && !r.startsWith("/"); i--) {
    if (parts[i]) r = r ? parts[i] + "/" + r : parts[i];
  }
  return normalize(r.startsWith("/") ? r : "/" + r);
};
const basename = (p, ext) => {
  const b = normalize(p).split("/").pop();
  return ext && b !== ext && b.endsWith(ext) ? b.slice(0, -ext.length) : b;
};
const path = {
  sep: "/",
  isAbsolute: (p) => p.startsWith("/"),
  normalize,
  join: (...parts) => normalize(parts.filter((s) => s !== "").join("/") || "."),
  resolve,
  dirname: (p) => {
    const n = normalize(p);
    const i = n.lastIndexOf("/");
    return i < 0 ? "." : i === 0 ? "/" : n.slice(0, i);
  },
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
const dirents = (entries, dir, opts) => opts?.withFileTypes
  ? entries.map((e) => ({
    name: e.name, parentPath: dir, path: dir,
    isFile: () => !e.isDir, isDirectory: () => e.isDir, isSymbolicLink: () => false,
  }))
  : entries.map((e) => e.name);
const globArgs = (pattern, opts) => ({
  pattern: stringArg(pattern, "pattern"),
  cwd: opts?.cwd == null ? "/" : pathArg(opts.cwd, "cwd"),
});
const globPaths = (r) => {
  if (r.truncated) globalThis.console?.warn("fs.glob: the search stopped early; results are incomplete");
  return r.paths;
};

const promises = {
  readFile: async (p, opts) => bufferLike(await acall("readFile", readArgs(p, opts))),
  readdir: async (p, opts) =>
    dirents(await acall("readdir", { path: pathArg(p) }), pathArg(p), opts),
  glob: async function* (pattern, opts) {
    yield* globPaths(await acall("glob", globArgs(pattern, opts)));
  },
};
const fs = {
  readFileSync: (p, opts) => bufferLike(call("readFile", readArgs(p, opts))),
  readdirSync: (p, opts) => dirents(call("readdir", { path: pathArg(p) }), pathArg(p), opts),
  existsSync: (p) => {
    try {
      const abs = pathArg(p);
      return abs === "/" || fs.readdirSync(path.dirname(abs)).includes(basename(abs));
    } catch (e) {
      // A host failure, such as the call budget, has no code, and `ENOSYS` means
      // there is no `ls` tool to ask. A missing parent throws `EIO`, because `ls`
      // reports no error code, so `EIO` means the path does not exist.
      if (e?.code === undefined || e.code === "ENOSYS") throw e;
      return false;
    }
  },
  globSync: (pattern, opts) => globPaths(call("glob", globArgs(pattern, opts))),
  promises,
};
const modules = { "node:fs": fs, "node:fs/promises": promises, "node:path": path };
const aliases = __ALIASES__;
// A top-level `function fs() {}` in code makes the global non-configurable;
// keep the code's own binding then rather than fail every later eval.
const define = (name, value) => {
  if (Object.getOwnPropertyDescriptor(globalThis, name)?.configurable === false) return;
  Object.defineProperty(globalThis, name, { value, writable: true, configurable: true, enumerable: false });
};
define("__lc_fs", { modules });
define("fs", fs);
define("require", (id) => {
  if (Object.prototype.hasOwnProperty.call(aliases, id)) return modules[aliases[id]];
  const err = new Error(`Cannot find module '${id}'`);
  err.code = "MODULE_NOT_FOUND";
  throw err;
});
})();
