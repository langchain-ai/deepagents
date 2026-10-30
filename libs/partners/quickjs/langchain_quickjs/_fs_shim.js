// Installs `fs` over the host functions in `_fs.py`. Runs before every eval,
// so it replaces an older copy that a restored snapshot carried.
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
const promises = {
  readFile: async (p, opts) => bufferLike(await acall("readFile", readArgs(p, opts))),
  readdir: async (p) => acall("readdir", { path: pathArg(p) }),
};
const fs = {
  readFileSync: (p, opts) => bufferLike(call("readFile", readArgs(p, opts))),
  readdirSync: (p) => call("readdir", { path: pathArg(p) }),
  promises,
};
// A top-level `function fs() {}` in code makes the global non-configurable;
// keep the code's own binding then rather than fail every later eval.
const define = (name, value) => {
  if (Object.getOwnPropertyDescriptor(globalThis, name)?.configurable === false) return;
  Object.defineProperty(globalThis, name, { value, writable: true, configurable: true, enumerable: false });
};
define("fs", fs);
})();
