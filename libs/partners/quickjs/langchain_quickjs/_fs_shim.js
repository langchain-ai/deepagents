((host) => {
const call = async (op, args) => {
  const { value, error } = await host(op, args);
  if (!error) return value;
  const { message, ...props } = error;
  throw Object.assign(new (props.name === "TypeError" ? TypeError : Error)(message), props);
};
const argError = (code, message) => Object.assign(new TypeError(message), { code });
const pathArg = (p) => {
  if (typeof p === "string") return p;
  throw argError("ERR_INVALID_ARG_TYPE", 'The "path" argument must be of type string');
};
const textOnly = (options) => {
  const encoding = typeof options === "string" ? options : options?.encoding;
  if (encoding == null || /^utf-?8$/i.test(encoding)) return;
  throw argError("ERR_INVALID_ARG_VALUE", `fs reads UTF-8 text only. Received encoding '${encoding}'`);
};
const promised = (fn) => async (...args) => {
  if (typeof args[args.length - 1] === "function") {
    throw argError("ERR_INVALID_ARG_TYPE", "fs functions return promises and take no callback; await the result instead");
  }
  return fn(...args);
};

const api = {
  readFile: promised(async (p, options) => {
    textOnly(options);
    return call("readFile", { path: pathArg(p) });
  }),
};
const instead = {
  readFileSync: "await fs.readFile(path)",
  writeFileSync: "the write_file tool",
  readdirSync: "the ls tool",
  existsSync: "the ls tool",
  exists: "the ls tool",
  accessSync: "the ls tool",
  stat: "the ls tool",
  statSync: "the ls tool",
  lstat: "the ls tool",
  glob: "the glob tool",
  rmSync: "the delete tool",
  unlink: "the delete tool",
  rmdir: "the delete tool",
  appendFile: "await fs.readFile(path), then the write_file tool",
  mkdir: "the write_file tool, which creates parent directories",
};
for (const [name, use] of Object.entries(instead)) {
  api[name] = () => {
    const message = `fs.${name} is not supported here, where fs is async and text-only. Use ${use}.`;
    throw Object.assign(new Error(message), { code: "ERR_METHOD_NOT_IMPLEMENTED" });
  };
}

// A top-level `function fs() {}` in code makes the global non-configurable;
// keep the code's own binding then rather than fail every later eval.
const define = (name, value) => {
  if (Object.getOwnPropertyDescriptor(globalThis, name)?.configurable === false) return;
  Object.defineProperty(globalThis, name, { value, writable: true, configurable: true, enumerable: false });
};
define("fs", { ...api, promises: { ...api } });
define("require", (id) => {
  const message = `Cannot find module '${id}': require is not supported here. fs is a global; use it directly.`;
  throw Object.assign(new Error(message), { code: "MODULE_NOT_FOUND" });
});
})
