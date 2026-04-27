export interface RuntimeConfigSupabase {
  auth: "supabase";
  supabaseUrl: string;
  supabaseAnonKey: string;
  appName: string;
  assistantId: string;
}

export interface RuntimeConfigClerk {
  auth: "clerk";
  clerkPublishableKey: string;
  appName: string;
  assistantId: string;
}

export interface RuntimeConfigAnonymous {
  auth: "anonymous";
  appName: string;
  assistantId: string;
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
  if (cfg.auth === "supabase") {
    if (!cfg.supabaseUrl || !cfg.supabaseAnonKey) {
      throw new Error("Runtime config missing supabaseUrl / supabaseAnonKey.");
    }
    return {
      auth: "supabase",
      supabaseUrl: cfg.supabaseUrl,
      supabaseAnonKey: cfg.supabaseAnonKey,
      appName: cfg.appName ?? "Deep Agent",
      assistantId: cfg.assistantId ?? "agent",
    };
  }
  if (cfg.auth === "clerk") {
    if (!cfg.clerkPublishableKey) {
      throw new Error("Runtime config missing clerkPublishableKey.");
    }
    return {
      auth: "clerk",
      clerkPublishableKey: cfg.clerkPublishableKey,
      appName: cfg.appName ?? "Deep Agent",
      assistantId: cfg.assistantId ?? "agent",
    };
  }
  if (cfg.auth === "anonymous") {
    return {
      auth: "anonymous",
      appName: cfg.appName ?? "Deep Agent",
      assistantId: cfg.assistantId ?? "agent",
    };
  }
  throw new Error(`Unknown auth provider: ${String(cfg.auth)}`);
}
