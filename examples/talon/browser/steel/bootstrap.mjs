import fs from "node:fs";

const profile = "/var/lib/steel/profile";
const proxy = "http://172.30.14.3:8080";
Object.assign(process.env, {
  CHROME_USER_DATA_DIR: profile,
  PROXY_URL: proxy,
  CHROME_HEADLESS: "true",
  CHROME_ARGS: "",
  FILTER_CHROME_ARGS: "",
  LOG_STORAGE_ENABLED: "false",
  ENABLE_CDP_LOGGING: "false",
  LOG_CUSTOM_EMIT_EVENTS: "false",
  DEBUG_CHROME_PROCESS: "false",
  ENABLE_VERBOSE_LOGGING: "false",
  SKIP_FINGERPRINT_INJECTION: "true",
  DEFAULT_TIMEZONE: "UTC",
});

const { default: puppeteer } = await import("puppeteer-core");
const { CDPService } = await import("../build/services/cdp/cdp.service.js");
const { SessionService } = await import("../build/services/session.service.js");
const { TargetInstrumentationManager } = await import("../build/services/cdp/instrumentation/target-manager.js");
const { loggingConfig } = await import("../build/config.js");
for (const key of Object.keys(loggingConfig)) loggingConfig[key] = false;

const failure = (code) => Object.assign(new Error(code), { code, statusCode: 400 });
let stopping = false;
let service;
let launching;
let closing;
const launchBrowser = puppeteer.launch.bind(puppeteer);
puppeteer.launch = async (options) => {
  if (stopping) throw failure("browser_stopping");
  fs.writeFileSync(`${profile}/.talon-dirty`, "active\n", { mode: 0o600, flag: "wx" });
  return launchBrowser({
    ...options,
    userDataDir: profile,
    executablePath: "/usr/lib/chromium/chromium",
    handleSIGTERM: false,
    handleSIGINT: false,
    handleSIGHUP: false,
    dumpio: false,
    args: options.args.filter((arg) => !/^--(load-extension|disable-extensions-except|proxy-server|proxy-bypass-list|unsafely-treat-insecure-origin-as-secure)/.test(arg)).concat([
      "--disable-extensions",
      `--proxy-server=${proxy}`,
      "--proxy-bypass-list=<-loopback>",
      "--disable-quic",
      "--disable-save-password-bubble",
    ]),
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
  const requested = config || this.defaultLaunchConfig;
  const selected = {
    ...requested,
    userDataDir: profile,
    userPreferences: undefined,
    extensions: [],
    options: { ...requested.options, headless: true, proxyUrl: proxy },
  };
  launching = launch.call(this, selected, { maxAttempts: 1 });
  return launching;
};
CDPService.prototype.shutdown = function () {
  if (closing) return closing;
  closing = (async () => {
    this.shuttingDown = true;
    if (this.browserInstance) {
      await this.browserInstance.close();
      this.browserInstance = null;
      fs.unlinkSync(`${profile}/.talon-dirty`);
    }
    this.wsEndpoint = null;
    this.currentSessionConfig = null;
  })().finally(() => { closing = undefined; });
  return closing;
};
CDPService.prototype.endSession = async function () {
  await this.shutdown();
};
CDPService.prototype.onDisconnect = async function () {
  if (!this.shuttingDown) {
    process.stderr.write('{"error":"browser_disconnected","profile":"dirty"}\n');
    process.exit(1);
  }
};
const start = SessionService.prototype.startSession;
SessionService.prototype.startSession = function (options = {}) {
  if (Object.entries(options).some(([key, value]) => value !== undefined && !["sessionId", "timeout", "blockAds", "isSelenium", "dimensions", "userAgent", "timezone"].includes(key)) || options.isSelenium === true) {
    throw failure("session_options_disabled");
  }
  return start.call(this, { ...options, timezone: options.timezone || "UTC" });
};

async function stop() {
  if (stopping) return;
  stopping = true;
  const timer = setTimeout(() => {
    process.stderr.write('{"error":"browser_shutdown_timeout","profile":"dirty"}\n');
    process.exit(1);
  }, 25000);
  try {
    await launching;
    await service?.shutdown();
    clearTimeout(timer);
    process.stdout.write('{"event":"browser_closed"}\n');
    process.exit(0);
  } catch {
    process.stderr.write('{"error":"browser_shutdown_failed","profile":"dirty"}\n');
    process.exit(1);
  }
}
process.on("SIGTERM", stop);
process.on("SIGINT", stop);
await import("../build/index.js");
