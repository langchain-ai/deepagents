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
  throw argError("ERR_INVALID_ARG_VALUE", `fs reads and writes UTF-8 text only. Received encoding '${encoding}'`);
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
  writeFile: promised(async (p, data, options) => {
    textOnly(options);
    if (![undefined, "w"].includes(options?.flag)) throw argError("ERR_INVALID_ARG_VALUE", `fs.writeFile creates or replaces a file only. Received flag '${options.flag}'`);
    if (typeof data !== "string") throw argError("ERR_INVALID_ARG_TYPE", 'The "data" argument must be of type string; fs writes text only');
    await call("writeFile", { path: pathArg(p), data });
  }),
  readdir: promised(async (p, options) => {
    textOnly(options);
    if (options?.withFileTypes || options?.recursive) throw argError("ERR_INVALID_ARG_VALUE", "fs.readdir lists the names in one directory only; withFileTypes and recursive are not supported");
    return call("readdir", { path: pathArg(p) });
  }),
  access: promised(async (p) => void (await call("access", { path: pathArg(p) }))),
  // The delete tool deletes a directory recursively, so check the path first, as Node does.
  rm: promised(async (p, options) => {
    const path = pathArg(p);
    let isDirectory;
    try {
      isDirectory = await call("rmStat", { path });
    } catch (e) {
      if (e.code === "ENOENT" && options?.force) return;
      throw e;
    }
    if (isDirectory && !options?.recursive) {
      const message = `Path is a directory: rm returned EISDIR (is a directory) ${path}`;
      throw Object.assign(new Error(message), { name: "SystemError", code: "ERR_FS_EISDIR", syscall: "rm", path });
    }
    await call("rm", { path });
  }),
  constants: { F_OK: 0, R_OK: 4, W_OK: 2, X_OK: 1 },
};
const instead = {
  readFileSync: "await fs.readFile(path)",
  writeFileSync: "await fs.writeFile(path, text)",
  readdirSync: "await fs.readdir(path)",
  existsSync: "await fs.access(path), which rejects with ENOENT for a missing path",
  exists: "await fs.access(path), which rejects with ENOENT for a missing path",
  accessSync: "await fs.access(path)",
  stat: "await fs.access(path) or await fs.readdir(dir)",
  statSync: "await fs.access(path) or await fs.readdir(dir)",
  lstat: "await fs.access(path) or await fs.readdir(dir)",
  glob: "the glob tool, or await fs.readdir(dir) for one directory",
  rmSync: "await fs.rm(path)",
  unlink: "await fs.rm(path)",
  rmdir: "await fs.rm(path, { recursive: true })",
  appendFile: "await fs.readFile(path), then await fs.writeFile(path, text)",
  mkdir: "nothing: await fs.writeFile(path, text) creates parent directories",
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
