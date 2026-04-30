import type { ReactNode } from "react";

import type { AuthAdapter, SessionState } from "./types";

const COOKIE_NAME = "dap_anon_id";

function readCookie(name: string): string | null {
  if (typeof document === "undefined") return null;
  const escaped = name.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const match = document.cookie.match(new RegExp(`(?:^|; )${escaped}=([^;]*)`));
  return match ? decodeURIComponent(match[1]) : null;
}

function writeCookie(name: string, value: string): void {
  if (typeof document === "undefined") return;
  // 1 year, path=/, SameSite=Lax. Not Secure (must work over http on localhost).
  // Not httpOnly — JS needs to read it.
  document.cookie = `${name}=${encodeURIComponent(value)}; max-age=31536000; path=/; SameSite=Lax`;
}

function getOrMintAnonId(): string {
  if (typeof crypto === "undefined") return "anon";
  const existing = readCookie(COOKIE_NAME);
  if (existing) return existing;
  const id = crypto.randomUUID();
  writeCookie(COOKIE_NAME, id);
  return id;
}

const session: SessionState = {
  status: "signed-in",
  accessToken: "",
  userIdentity: getOrMintAnonId(),
  userEmail: null,
  isAnonymous: true,
  signOut: async () => {},
};

const adapter: AuthAdapter = {
  Provider: ({ children }: { children: ReactNode }) => <>{children}</>,
  useSession: () => session,
  AuthUI: () => null,
};

export default adapter;
