import {
  ClerkProvider,
  SignIn,
  SignUp,
  useAuth,
  useUser,
} from "@clerk/clerk-react";
import { dark } from "@clerk/themes";
import {
  createContext,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";

import { getRuntimeConfig } from "../runtimeConfig";
import { useTheme } from "../ThemeProvider";
import type { AuthAdapter, SessionState } from "./types";

type Ctx = { state: SessionState };
const ClerkCtx = createContext<Ctx | null>(null);

function ClerkSessionBridge({ children }: { children: ReactNode }) {
  const { isLoaded, isSignedIn, getToken, signOut } = useAuth();
  const { user } = useUser();
  const [accessToken, setAccessToken] = useState<string | null>(null);

  const userEmail = user?.primaryEmailAddress?.emailAddress ?? null;

  useEffect(() => {
    let active = true;
    if (!isLoaded || !isSignedIn) {
      setAccessToken(null);
      return;
    }
    // Clerk session JWTs default to a ~60s TTL; refresh well under that so the
    // LangGraph SDK never gets handed an expired token.
    const refresh = async () => {
      try {
        const t = await getToken({ skipCache: true });
        if (active && t) setAccessToken(t);
      } catch (err) {
        if (active) console.warn("Clerk getToken failed", err);
      }
    };
    void refresh();
    const interval = window.setInterval(refresh, 45_000);
    return () => {
      active = false;
      window.clearInterval(interval);
    };
  }, [getToken, isLoaded, isSignedIn, user?.id]);

  const value = useMemo<Ctx>(() => {
    let state: SessionState;
    if (!isLoaded) state = { status: "loading" };
    else if (!isSignedIn) state = { status: "signed-out" };
    else if (!accessToken) state = { status: "loading" };
    else {
      state = {
        status: "signed-in",
        accessToken,
        userIdentity: user?.id ?? "",
        userEmail,
        signOut: async () => {
          await signOut();
        },
      };
    }
    return { state };
  }, [isLoaded, isSignedIn, accessToken, user?.id, userEmail, signOut]);

  return <ClerkCtx.Provider value={value}>{children}</ClerkCtx.Provider>;
}

function ClerkAdapterProvider({ children }: { children: ReactNode }) {
  const cfg = getRuntimeConfig();
  const { theme } = useTheme();
  if (cfg.auth !== "clerk") {
    throw new Error("ClerkProvider mounted with non-clerk runtime config");
  }
  return (
    <ClerkProvider
      publishableKey={cfg.clerkPublishableKey}
      appearance={theme === "dark" ? { baseTheme: dark } : undefined}
      // The SPA is mounted at /app/; Clerk's default of "/" lands on the
      // LangGraph root (404/JSON). Pin post-auth redirects back into the app.
      afterSignOutUrl="/app/"
      signInFallbackRedirectUrl="/app/"
      signUpFallbackRedirectUrl="/app/"
    >
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
    <div className="min-h-dvh flex items-center justify-center bg-[var(--background)] p-4">
      <div className="flex flex-col items-center gap-4">
        {mode === "signin" ? (
          <SignIn routing="virtual" />
        ) : (
          <SignUp routing="virtual" />
        )}
        <button
          type="button"
          className="text-xs text-[var(--muted-foreground)] hover:underline"
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
