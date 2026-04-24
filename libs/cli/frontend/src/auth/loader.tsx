import { use } from "react";

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
 *
 * The result is memoized at module scope — safe for repeated calls (which
 * `use()` may do during render-pass retries).
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
 * Returns the resolved auth adapter. Suspends the caller until the adapter
 * module has loaded — use inside a `<Suspense>` boundary.
 *
 * Prefer this over a `useState + useEffect + null-check` pattern: that's
 * the anti-pattern called out in
 * https://react.dev/learn/you-might-not-need-an-effect — the React 19
 * `use()` hook handles async resources natively.
 */
export function useAuthAdapter(): AuthAdapter {
  return use(loadAuth());
}
