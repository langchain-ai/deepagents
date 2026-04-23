import { useEffect, useMemo, useState, type ReactNode } from "react";

import { getRuntimeConfig } from "../runtimeConfig";
import type { AuthAdapter } from "./types";

let _cache: Promise<AuthAdapter> | null = null;

/**
 * Dynamic-import the auth adapter module for the active provider.
 *
 * Vite code-splits each adapter (and its transitive SDK) into its own chunk.
 * At runtime, only the chunk matching `runtimeConfig.auth` is fetched — so
 * the other SDK's code is shipped in the dist folder but never downloaded
 * by the user's browser.
 */
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

/**
 * React hook that returns the adapter once its module has loaded.
 * Returns `null` during the initial fetch.
 */
export function useAuthAdapter(): AuthAdapter | null {
  const promise = useMemo(() => loadAuth(), []);
  const [adapter, setAdapter] = useState<AuthAdapter | null>(null);
  useEffect(() => {
    let active = true;
    void promise.then((a) => {
      if (active) setAdapter(a);
    });
    return () => {
      active = false;
    };
  }, [promise]);
  return adapter;
}

// Parameter `children` is typed for callers that use a render-prop pattern.
export interface AuthLoaderRenderProp {
  (adapter: AuthAdapter): ReactNode;
}
