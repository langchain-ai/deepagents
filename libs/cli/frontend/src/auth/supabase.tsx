import { Auth } from "@supabase/auth-ui-react";
import { ThemeSupa } from "@supabase/auth-ui-shared";
import { createClient, type Session, type SupabaseClient } from "@supabase/supabase-js";
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

type Ctx = {
  supabase: SupabaseClient;
  state: SessionState;
  // While recoveryMode is true, `state.status` is forced to "signed-out"
  // so the update-password view stays mounted even if Supabase has minted
  // a short-lived session from the recovery token.
  recoveryMode: boolean;
  clearRecoveryMode: () => void;
};

const SupabaseCtx = createContext<Ctx | null>(null);

function detectRecoveryInUrl(): boolean {
  if (typeof window === "undefined") return false;
  // Supabase versions differ between `type=recovery` and `recovery_type=`
  // in the URL fragment — accept either.
  const hash = window.location.hash;
  return hash.includes("type=recovery") || hash.includes("recovery_type=");
}

function SupabaseProvider({ children }: { children: ReactNode }) {
  const cfg = getRuntimeConfig();
  if (cfg.auth !== "supabase") {
    throw new Error("SupabaseProvider mounted with non-supabase runtime config");
  }

  const supabase = useMemo(
    () => createClient(cfg.supabaseUrl, cfg.supabaseAnonKey),
    [cfg.supabaseUrl, cfg.supabaseAnonKey],
  );
  const [session, setSession] = useState<Session | null>(null);
  const [loading, setLoading] = useState(true);
  // Must be set before any effect runs: `createClient({ detectSessionInUrl: true })`
  // parses the hash synchronously and emits PASSWORD_RECOVERY before any
  // listener can attach. Detecting from the URL directly is the only reliable path.
  const [recoveryMode, setRecoveryMode] = useState<boolean>(detectRecoveryInUrl);

  useEffect(() => {
    let active = true;
    supabase.auth
      .getSession()
      .then(({ data }) => {
        if (!active) return;
        setSession(data.session);
        setLoading(false);
      })
      .catch((err) => {
        if (!active) return;
        console.error("Failed to load Supabase session", err);
        setLoading(false);
      });
    const { data: { subscription } } = supabase.auth.onAuthStateChange(
      (event, nextSession) => {
        if (!active) return;
        setSession(nextSession);
        setLoading(false);
        // Cross-tab: another tab enters recovery, mirror that here.
        if (event === "PASSWORD_RECOVERY") {
          setRecoveryMode(true);
        }
        // updateUser({ password }) succeeded — recovery flow is complete.
        if (event === "USER_UPDATED") {
          setRecoveryMode(false);
        }
      },
    );
    return () => {
      active = false;
      subscription.unsubscribe();
    };
  }, [supabase]);

  const baseState: SessionState = loading
    ? { status: "loading" }
    : session
      ? {
          status: "signed-in",
          accessToken: session.access_token,
          userIdentity: session.user.id,
          userEmail: session.user.email ?? null,
          isAnonymous: false,
          signOut: async () => {
            await supabase.auth.signOut();
          },
        }
      : { status: "signed-out" };

  const state: SessionState = recoveryMode
    ? { status: "signed-out" }
    : baseState;

  const value = useMemo<Ctx>(
    () => ({
      supabase,
      state,
      recoveryMode,
      clearRecoveryMode: () => setRecoveryMode(false),
    }),
    [supabase, state, recoveryMode],
  );

  return <SupabaseCtx.Provider value={value}>{children}</SupabaseCtx.Provider>;
}

function useSupabaseCtx(): Ctx {
  const ctx = useContext(SupabaseCtx);
  if (!ctx) {
    throw new Error("useSession() called outside SupabaseProvider");
  }
  return ctx;
}

function useSession(): SessionState {
  return useSupabaseCtx().state;
}

function SupabaseAuthUI() {
  const { supabase, recoveryMode, clearRecoveryMode } = useSupabaseCtx();
  const { theme } = useTheme();

  // Strip the recovery hash once recovery completes so a refresh doesn't
  // re-enter recovery mode.
  useEffect(() => {
    if (!recoveryMode) {
      if (typeof window !== "undefined" && detectRecoveryInUrl()) {
        window.history.replaceState(null, "", window.location.pathname + window.location.search);
      }
    }
  }, [recoveryMode]);

  return (
    <div className="min-h-dvh flex items-center justify-center bg-[var(--background)] p-4">
      <div className="flex w-full max-w-sm flex-col gap-4 rounded-xl border border-[var(--border)] bg-[var(--surface)] p-6 shadow-sm">
        <div className="flex items-center gap-2">
          <img
            src={theme === "dark" ? "/app/logo-dark.svg" : "/app/logo-light.svg"}
            alt="Deep Agents"
            className="h-8 w-8 rounded"
          />
        </div>
        <Auth
          supabaseClient={supabase}
          view={recoveryMode ? "update_password" : "sign_in"}
          appearance={{
            theme: ThemeSupa,
            variables: {
              default: {
                colors: {
                  brand: "#7fc8ff",
                  brandAccent: "#99d4ff",
                  brandButtonText: "#030710",
                  defaultButtonBackground: "#ffffff",
                  defaultButtonBackgroundHover: "#f2faff",
                  defaultButtonBorder: "#b8dfff",
                  defaultButtonText: "#030710",
                  dividerBackground: "#e5f4ff",
                  inputBackground: "#ffffff",
                  inputBorder: "#b8dfff",
                  inputBorderHover: "#99d4ff",
                  inputBorderFocus: "#7fc8ff",
                  inputText: "#030710",
                  inputLabelText: "#030710",
                  inputPlaceholder: "#6b8299",
                  messageText: "#030710",
                  messageTextDanger: "#dc2626",
                  anchorTextColor: "#6b8299",
                  anchorTextHoverColor: "#030710",
                },
              },
              dark: {
                colors: {
                  brand: "#7fc8ff",
                  brandAccent: "#99d4ff",
                  brandButtonText: "#030710",
                  defaultButtonBackground: "#0a1220",
                  defaultButtonBackgroundHover: "#111a2b",
                  defaultButtonBorder: "#1f3a5c",
                  defaultButtonText: "#fafcff",
                  dividerBackground: "#1f3a5c",
                  inputBackground: "#0a1220",
                  inputBorder: "#1f3a5c",
                  inputBorderHover: "#7fc8ff",
                  inputBorderFocus: "#7fc8ff",
                  inputText: "#fafcff",
                  inputLabelText: "#fafcff",
                  inputPlaceholder: "#8a9aac",
                  messageText: "#fafcff",
                  messageTextDanger: "#f87171",
                  anchorTextColor: "#8a9aac",
                  anchorTextHoverColor: "#fafcff",
                },
              },
            },
          }}
          providers={[]}
          theme={theme}
          redirectTo={window.location.origin + "/app/"}
          showLinks={!recoveryMode}
        />
        {recoveryMode && (
          <button
            type="button"
            onClick={clearRecoveryMode}
            className="text-center text-xs text-[var(--muted-foreground)] hover:underline"
          >
            Cancel — back to sign in
          </button>
        )}
      </div>
    </div>
  );
}

const adapter: AuthAdapter = {
  Provider: SupabaseProvider,
  useSession,
  AuthUI: SupabaseAuthUI,
};

export default adapter;
