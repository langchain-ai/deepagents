// Installs `fs` before every eval, replacing a restored snapshot's copy. It takes
// its host functions as arguments, so code that reassigns globals cannot intercept.
((settle, fsSync, fsAsync, admit) => {
const call = (op, args) => settle(fsSync(op, args));
const acall = async (op, args) => (admit(), settle(await fsAsync(op, args)));

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

// The file tools resolve a relative path against `/`.
const pathArg = (p) => {
  if (typeof p === "string") return p;
  const err = new TypeError('The "path" argument must be of type string');
  err.code = "ERR_INVALID_ARG_TYPE";
  throw err;
};
const readArgs = (p, opts) => {
  const enc = typeof opts === "string" ? opts : opts?.encoding;
  return { path: pathArg(p), encoding: enc == null ? null : String(enc).toLowerCase() };
};

const promises = {
  readFile: async (p, opts) => bufferLike(await acall("readFile", readArgs(p, opts))),
};
const fs = {
  readFileSync: (p, opts) => bufferLike(call("readFile", readArgs(p, opts))),
  promises,
};
// A top-level `function fs() {}` in code makes the global non-configurable;
// keep the code's own binding then rather than fail every later eval.
if (Object.getOwnPropertyDescriptor(globalThis, "fs")?.configurable !== false) {
  Object.defineProperty(globalThis, "fs", { value: fs, writable: true, configurable: true, enumerable: false });
}
})
