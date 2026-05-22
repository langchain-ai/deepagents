import { mapWithConcurrency } from "./utils.ts";

export type StateFilter = "open" | "closed" | "all";

export type RepoScope = {
  org: string;
  repo: string;
  state?: StateFilter;
};

export type SourceItem = {
  id: number;
  type: "issue" | "pr" | "discussion";
  state: string;
  title: string;
  url: string;
  labels: string[];
  updated_at: string;
  body: string;
  comments: number;
};

export type FetchRunOptions = {
  fetch_concurrency?: number;
  page_delay_ms?: number;
  max_retries?: number;
  retry_base_delay_ms?: number;
};

const GH_API_BASE = "https://api.github.com";
const DEFAULT_STATE: StateFilter = "open";
const PER_PAGE = 100;
const MAX_PAGES = 100;
const DEFAULT_FETCH_CONCURRENCY = 2;
const DEFAULT_PAGE_DELAY_MS = 150;
const DEFAULT_MAX_RETRIES = 5;
const DEFAULT_RETRY_BASE_DELAY_MS = 800;

function sleep(ms: number): Promise<void> {
  if (ms <= 0) {
    return Promise.resolve();
  }
  const maybeSetTimeout = (globalThis as any).setTimeout;
  if (typeof maybeSetTimeout !== "function") {
    // QuickJS environments may not provide timers; skip delay in that case.
    return Promise.resolve();
  }
  return new Promise((resolve) => maybeSetTimeout(resolve, ms));
}

function jitter(ms: number): number {
  return Math.floor(ms * (1 + Math.random() * 0.2));
}

function isRateLimitLike(message: string): boolean {
  const haystack = message.toLowerCase();
  return (
    haystack.includes("429") ||
    haystack.includes("rate limit") ||
    haystack.includes("secondary rate limit") ||
    haystack.includes("abuse detection") ||
    haystack.includes("http 403")
  );
}

function toErrorMessage(err: unknown): string {
  if (err instanceof Error) {
    return err.message;
  }
  return String(err ?? "unknown error");
}

function safeParseJson(raw: string): any {
  const trimmed = raw.trim();
  if (!trimmed.startsWith("{") && !trimmed.startsWith("[")) {
    throw new Error(`Non-JSON fetch payload: ${trimmed.slice(0, 200)}`);
  }
  return JSON.parse(trimmed);
}

function normalizeLabels(rawLabels: any): string[] {
  if (!Array.isArray(rawLabels)) {
    return [];
  }
  return rawLabels
    .map((label: any) => {
      if (typeof label === "string") {
        return label;
      }
      if (label && typeof label.name === "string") {
        return label.name;
      }
      return "";
    })
    .filter(Boolean);
}

function q(params: Record<string, string>): string {
  return Object.entries(params)
    .map(([k, v]) => `${encodeURIComponent(k)}=${encodeURIComponent(v)}`)
    .join("&");
}

async function fetchJson(url: string, options: FetchRunOptions = {}): Promise<any> {
  const maxRetries = options.max_retries ?? DEFAULT_MAX_RETRIES;
  const baseDelayMs = options.retry_base_delay_ms ?? DEFAULT_RETRY_BASE_DELAY_MS;
  const fetchTool = (globalThis as any).tools?.fetch;
  if (!fetchTool) {
    throw new Error("fetch tool is unavailable");
  }

  for (let attempt = 0; attempt < maxRetries; attempt += 1) {
    try {
      const maybeObj = await fetchTool({ url });
      const raw = typeof maybeObj === "string" ? maybeObj : String(maybeObj ?? "");
      return safeParseJson(raw);
    } catch (err) {
      const message = toErrorMessage(err);
      const lastAttempt = attempt >= maxRetries - 1;
      if (lastAttempt || !isRateLimitLike(message)) {
        throw err;
      }
      const delayMs = jitter(baseDelayMs * 2 ** attempt);
      await sleep(delayMs);
    }
  }
  throw new Error("fetchJson retry loop exhausted");
}

async function paginate(
  pathBuilder: (page: number) => string,
  options: FetchRunOptions = {}
): Promise<any[]> {
  const pageDelayMs = options.page_delay_ms ?? DEFAULT_PAGE_DELAY_MS;
  const rows: any[] = [];
  for (let page = 1; page <= MAX_PAGES; page += 1) {
    const pageRows = await fetchJson(pathBuilder(page), options);
    if (!Array.isArray(pageRows) || pageRows.length === 0) {
      break;
    }
    rows.push(...pageRows);
    if (pageRows.length < PER_PAGE) {
      break;
    }
    if (pageDelayMs > 0) {
      await sleep(pageDelayMs);
    }
  }
  return rows;
}

export async function listIssues(
  scope: RepoScope,
  options: FetchRunOptions = {}
): Promise<SourceItem[]> {
  const state = scope.state ?? DEFAULT_STATE;
  const rows = await paginate((page) => {
    const params = q({
      state,
      per_page: String(PER_PAGE),
      page: String(page),
      sort: "updated",
      direction: "desc",
    });
    return `${GH_API_BASE}/repos/${scope.org}/${scope.repo}/issues?${params}`;
  }, options);

  return rows
    .filter((row: any) => !row.pull_request)
    .map((row: any) => ({
      id: Number(row.number),
      type: "issue" as const,
      state: String(row.state ?? ""),
      title: String(row.title ?? ""),
      url: String(row.html_url ?? ""),
      labels: normalizeLabels(row.labels),
      updated_at: String(row.updated_at ?? ""),
      body: String(row.body ?? ""),
      comments: Number(row.comments ?? 0),
    }));
}

export async function listPullRequests(
  scope: RepoScope,
  options: FetchRunOptions = {}
): Promise<SourceItem[]> {
  const state = scope.state ?? DEFAULT_STATE;
  const rows = await paginate((page) => {
    const params = q({
      state,
      per_page: String(PER_PAGE),
      page: String(page),
      sort: "updated",
      direction: "desc",
    });
    return `${GH_API_BASE}/repos/${scope.org}/${scope.repo}/pulls?${params}`;
  }, options);

  return rows.map((row: any) => ({
    id: Number(row.number),
    type: "pr" as const,
    state: String(row.state ?? ""),
    title: String(row.title ?? ""),
    url: String(row.html_url ?? ""),
    labels: normalizeLabels(row.labels),
    updated_at: String(row.updated_at ?? ""),
    body: String(row.body ?? ""),
    comments: Number(row.comments ?? 0),
  }));
}

export async function listDiscussions(
  scope: RepoScope,
  options: FetchRunOptions = {}
): Promise<SourceItem[]> {
  const state = scope.state ?? DEFAULT_STATE;
  const rows = await paginate((page) => {
    const params = q({
      per_page: String(PER_PAGE),
      page: String(page),
    });
    return `${GH_API_BASE}/repos/${scope.org}/${scope.repo}/discussions?${params}`;
  }, options);

  return rows
    .filter((row: any) => {
      if (state === "all") {
        return true;
      }
      return String(row.state ?? "").toLowerCase() === state;
    })
    .map((row: any) => ({
      id: Number(row.number),
      type: "discussion" as const,
      state: String(row.state ?? ""),
      title: String(row.title ?? ""),
      url: String(row.html_url ?? ""),
      labels: normalizeLabels(row.labels),
      updated_at: String(row.updated_at ?? ""),
      body: String(row.body ?? ""),
      comments: Number(row.comments ?? 0),
    }));
}

export async function listRepoItems(
  scope: RepoScope,
  options: FetchRunOptions = {}
): Promise<SourceItem[]> {
  const fetchConcurrency = options.fetch_concurrency ?? DEFAULT_FETCH_CONCURRENCY;
  const loaders = [
    () => listIssues(scope, options),
    () => listPullRequests(scope, options),
    () => listDiscussions(scope, options),
  ];
  const [issues, prs, discussions] = await mapWithConcurrency(
    loaders,
    fetchConcurrency,
    (fn) => fn()
  );
  return [...issues, ...prs, ...discussions].sort((a, b) => a.id - b.id);
}
