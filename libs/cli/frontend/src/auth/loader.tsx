import { use } from "react";

import { getRuntimeConfig } from "../runtimeConfig";
import type { AuthAdapter } from "./types";

// Module-scope cache so `use()` retries during a render pass return the same promise.
let _cache: Promise<AuthAdapter> | null = null;

export function loadAuth(): Promise<AuthAdapter> {
  if (_cache) return _cache;
  const cfg = getRuntimeConfig();
  _cache = (async () => {
    const mod =
      cfg.auth === "supabase"
        ? await import("./supabase")
        : await import("./clerk");
    return mod.default;
  })();
  return _cache;
}

export function useAuthAdapter(): AuthAdapter {
  return use(loadAuth());
}
