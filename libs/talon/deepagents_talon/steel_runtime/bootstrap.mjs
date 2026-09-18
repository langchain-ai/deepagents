import childProcess from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { createRequire, syncBuiltinESMExports } from "node:module";
import { pathToFileURL } from "node:url";

const source = process.argv[2];
const require = createRequire(path.join(source, "package.json"));
const upstream = (name) => import(pathToFileURL(path.join(source, "api/build", name)));
const dependency = (name) => import(pathToFileURL(require.resolve(name)));
const profile = process.env.CHROME_USER_DATA_DIR;
const dirty = path.join(profile, ".talon-dirty");
const spawn = childProcess.spawn;
// Puppeteer otherwise detaches Chrome, defeating the host's process-group cleanup.
childProcess.spawn = function (command, args, options) {
  return spawn(command, args, command === process.env.CHROME_EXECUTABLE_PATH
    ? { ...options, detached: false } : options);
};
syncBuiltinESMExports();

// Match the ESM instance imported by Steel, not Puppeteer's separate CJS export.
const { default: puppeteer } = await import(pathToFileURL(path.join(
  source, "node_modules/puppeteer-core/lib/esm/puppeteer/puppeteer-core.js",
)));
const { default: fastify } = await dependency("fastify");
const { CDPService } = await upstream("services/cdp/cdp.service.js");
const { SessionService } = await upstream("services/session.service.js");
const { FileService } = await upstream("services/file.service.js");
const { TargetInstrumentationManager } = await upstream("services/cdp/instrumentation/target-manager.js");
const files = new FileService({
  baseFilesPath: path.join(process.cwd(), "files"),
  prebuiltArchiveDir: path.join(process.cwd(), "archive"),
  watchFiles: false,
});
FileService.getInstance = () => files;

const failure = (code) => Object.assign(new Error(code), { code, statusCode: 400 });
let stopping = false;
let service;
let bridge;
let launching;
let closing;
let markLaunch;
const firstLaunch = new Promise((resolve) => { markLaunch = resolve; });
const connect = puppeteer.connect.bind(puppeteer);
puppeteer.connect = (options) => connect({
  ...options,
  browserWSEndpoint: options.browserWSEndpoint === `ws://127.0.0.1:${process.env.PORT}`
    ? service?.wsEndpoint : options.browserWSEndpoint,
});
const launchBrowser = puppeteer.launch.bind(puppeteer);
puppeteer.launch = async (options) => {
  if (stopping) throw failure("browser_stopping");
  fs.writeFileSync(dirty, "active\n", { mode: 0o600, flag: "wx" });
  return launchBrowser({
    ...options,
    handleSIGTERM: false,
    handleSIGINT: false,
    handleSIGHUP: false,
    args: options.args.filter((arg) => !/^--(load-extension|disable-extensions-except|unsafely-treat-insecure-origin-as-secure)/.test(arg)).concat("--disable-extensions"),
  });
};
TargetInstrumentationManager.prototype.attach = async function () {};
CDPService.prototype.setupUserPreferences = async function () {};
CDPService.prototype.getSessionData = async function () { throw failure("profile_export_disabled"); };
CDPService.prototype.getBrowserState = async function () { throw failure("profile_export_disabled"); };
const launch = CDPService.prototype.launch;
CDPService.prototype.launch = function (config) {
  if (stopping) return Promise.reject(failure("browser_stopping"));
  service = this;
  this.shuttingDown = false;
  this.instrumentationLogger.record = () => {};
  launching = launch.call(this, {
    ...(config || this.defaultLaunchConfig),
    userDataDir: profile,
    userPreferences: undefined,
    extensions: [],
  }, { maxAttempts: 1 });
  markLaunch(launching);
  return launching;
};
CDPService.prototype.shutdown = function () {
  if (closing) return closing;
  closing = (async () => {
    this.shuttingDown = true;
    if (this.browserInstance) {
      await this.browserInstance.close();
      this.browserInstance = null;
      fs.unlinkSync(dirty);
    }
    this.wsEndpoint = null;
    this.currentSessionConfig = null;
  })().finally(() => { closing = undefined; });
  return closing;
};
CDPService.prototype.endSession = async function () { await this.shutdown(); };
CDPService.prototype.onDisconnect = async function () {
  if (!this.shuttingDown) process.exit(1);
};
const startSession = SessionService.prototype.startSession;
SessionService.prototype.startSession = function (options = {}) {
  const allowed = ["sessionId", "timeout", "blockAds", "isSelenium", "dimensions", "userAgent", "timezone"];
  if (Object.entries(options).some(([key, value]) => value !== undefined && !allowed.includes(key)) || options.isSelenium === true) {
    throw failure("session_options_disabled");
  }
  return startSession.call(this, { ...options, timezone: options.timezone || "UTC" });
};

async function stop(exitCode = 0) {
  if (stopping) return;
  stopping = true;
  setTimeout(() => process.exit(1), 25000);
  try {
    await launching?.catch(() => {});
    await bridge?.close();
    await service?.shutdown();
    process.exit(exitCode);
  } catch {
    process.exit(1);
  }
}
process.on("SIGTERM", () => stop());
process.on("SIGINT", () => stop());

const { default: steel } = await upstream("steel-browser-plugin.js");
const server = fastify({ logger: false, bodyLimit: 1024 * 1024 });
const authority = `127.0.0.1:${process.env.PORT}`;
const localRequest = (request) => request.headers.host === authority &&
  (!request.headers.origin || request.headers.origin === `http://${authority}`);
server.addHook("onRequest", async (request, reply) => {
  if (!localRequest(request)) {
    return reply.code(403).send({ error: "local_browser_only" });
  }
});
await server.register(steel, {
  logging: { enableStorage: false, enableConsoleLogging: false, enableLogsRoutes: false },
});
server.server.prependListener("upgrade", (request, socket) => {
  if (!localRequest(request)) socket.destroy();
});
try {
  await server.listen({ host: "127.0.0.1", port: Number(process.env.PORT) });
  await firstLaunch;
  if (!service?.wsEndpoint) throw failure("browser_not_ready");
  if (process.env.TALON_BROWSER_IDENTITIES) {
    const { main } = await import('./bridge.mjs');
    bridge = await main();
  }
  process.stdout.write('{"event":"talon_steel_ready"}\n');
} catch {
  await stop(1);
}
