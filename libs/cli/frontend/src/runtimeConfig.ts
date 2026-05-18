interface RuntimeConfigBase {
  appName: string;
  assistantId: string;
  subtitle?: string;
  prompts?: string[];
}

export interface RuntimeConfigSupabase extends RuntimeConfigBase {
  auth: "supabase";
  supabaseUrl: string;
  supabaseAnonKey: string;
}

export interface RuntimeConfigClerk extends RuntimeConfigBase {
  auth: "clerk";
  clerkPublishableKey: string;
}

export interface RuntimeConfigAnonymous extends RuntimeConfigBase {
  auth: "anonymous";
}

export type RuntimeConfig =
  | RuntimeConfigSupabase
  | RuntimeConfigClerk
  | RuntimeConfigAnonymous;

declare global {
  interface Window {
    __DEEPAGENTS_CONFIG__?: Partial<RuntimeConfig> & { __PLACEHOLDER__?: boolean };
  }
}

export function getRuntimeConfig(): RuntimeConfig {
  const cfg = window.__DEEPAGENTS_CONFIG__;
  if (!cfg || cfg.__PLACEHOLDER__) {
    throw new Error(
      "window.__DEEPAGENTS_CONFIG__ not injected. Run through `deepagent deploy` or `deepagent dev`.",
    );
  }
  const base = {
    appName: cfg.appName ?? "Deep Agent",
    assistantId: cfg.assistantId ?? "agent",
    subtitle: cfg.subtitle,
    prompts: cfg.prompts,
  };
  if (cfg.auth === "supabase") {
    if (!cfg.supabaseUrl || !cfg.supabaseAnonKey) {
      throw new Error("Runtime config missing supabaseUrl / supabaseAnonKey.");
    }
    return {
      auth: "supabase",
      supabaseUrl: cfg.supabaseUrl,
      supabaseAnonKey: cfg.supabaseAnonKey,
      ...base,
    };
  }
  if (cfg.auth === "clerk") {
    if (!cfg.clerkPublishableKey) {
      throw new Error("Runtime config missing clerkPublishableKey.");
    }
    return {
      auth: "clerk",
      clerkPublishableKey: cfg.clerkPublishableKey,
      ...base,
    };
  }
  if (cfg.auth === "anonymous") {
    return {
      auth: "anonymous",
      ...base,
    };
  }
  throw new Error(`Unknown auth provider: ${String(cfg.auth)}`);
}
