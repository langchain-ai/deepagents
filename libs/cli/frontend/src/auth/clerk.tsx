import {
  ClerkProvider,
  SignIn,
  SignUp,
  useAuth,
  useUser,
} from "@clerk/clerk-react";
import {
  createContext,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";

import { getRuntimeConfig } from "../runtimeConfig";
import type { AuthAdapter, SessionState } from "./types";

type Ctx = { state: SessionState };
const ClerkCtx = createContext<Ctx | null>(null);

function ClerkSessionBridge({ children }: { children: ReactNode }) {
  const { isLoaded, isSignedIn, getToken, signOut } = useAuth();
  const { user } = useUser();
  const [accessToken, setAccessToken] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    if (!isLoaded || !isSignedIn) {
      setAccessToken(null);
      return;
    }
    const refresh = async () => {
      // Clerk session tokens default to a ~60s TTL. Force a fresh fetch on an
      // interval well under that so long-idle sessions don't hand the
      // LangGraph SDK an expired JWT.
      const t = await getToken({ skipCache: true });
      if (active && t) setAccessToken(t);
    };
    void refresh();
    const interval = window.setInterval(refresh, 45_000);
    return () => {
      active = false;
      window.clearInterval(interval);
    };
  }, [getToken, isLoaded, isSignedIn, user?.id]);

  const state: SessionState = useMemo(() => {
    if (!isLoaded) return { status: "loading" };
    if (!isSignedIn) return { status: "signed-out" };
    if (!accessToken) return { status: "loading" };
    return {
      status: "signed-in",
      accessToken,
      userIdentity: user?.id ?? "",
      userEmail: user?.primaryEmailAddress?.emailAddress ?? null,
      signOut: async () => {
        await signOut();
      },
    };
  }, [isLoaded, isSignedIn, accessToken, user?.id, user?.primaryEmailAddress, signOut]);

  return <ClerkCtx.Provider value={{ state }}>{children}</ClerkCtx.Provider>;
}

function ClerkAdapterProvider({ children }: { children: ReactNode }) {
  const cfg = getRuntimeConfig();
  if (cfg.auth !== "clerk") {
    throw new Error("ClerkProvider mounted with non-clerk runtime config");
  }
  return (
    <ClerkProvider publishableKey={cfg.clerkPublishableKey}>
      <ClerkSessionBridge>{children}</ClerkSessionBridge>
    </ClerkProvider>
  );
}

function useSession(): SessionState {
  const ctx = useContext(ClerkCtx);
  if (!ctx) {
    throw new Error("useSession() called outside ClerkAdapterProvider");
  }
  return ctx.state;
}

function ClerkAuthUI() {
  const [mode, setMode] = useState<"signin" | "signup">("signin");
  return (
    <div className="min-h-dvh flex items-center justify-center bg-slate-50 p-4">
      <div className="flex flex-col items-center gap-4">
        {mode === "signin" ? (
          <SignIn routing="virtual" />
        ) : (
          <SignUp routing="virtual" />
        )}
        <button
          type="button"
          className="text-xs text-slate-600 hover:underline"
          onClick={() => setMode(mode === "signin" ? "signup" : "signin")}
        >
          {mode === "signin"
            ? "Need an account? Sign up"
            : "Already have an account? Sign in"}
        </button>
      </div>
    </div>
  );
}

const adapter: AuthAdapter = {
  Provider: ClerkAdapterProvider,
  useSession,
  AuthUI: ClerkAuthUI,
};

export default adapter;
