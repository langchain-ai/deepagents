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
};

const SupabaseCtx = createContext<Ctx | null>(null);

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

  useEffect(() => {
    let active = true;
    supabase.auth.getSession().then(({ data }) => {
      if (!active) return;
      setSession(data.session);
      setLoading(false);
    });
    const { data: { subscription } } = supabase.auth.onAuthStateChange(
      (_event, nextSession) => {
        if (!active) return;
        setSession(nextSession);
        setLoading(false);
      },
    );
    return () => {
      active = false;
      subscription.unsubscribe();
    };
  }, [supabase]);

  const state: SessionState = loading
    ? { status: "loading" }
    : session
      ? {
          status: "signed-in",
          accessToken: session.access_token,
          userIdentity: session.user.id,
          userEmail: session.user.email ?? null,
          signOut: async () => {
            await supabase.auth.signOut();
          },
        }
      : { status: "signed-out" };

  const value = useMemo<Ctx>(() => ({ supabase, state }), [supabase, state]);

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
  const { supabase } = useSupabaseCtx();
  const { theme } = useTheme();
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
          appearance={{
            theme: ThemeSupa,
            variables: {
              default: {
                colors: {
                  brand: "#7fc8ff",
                  brandAccent: "#99d4ff",
                },
              },
            },
          }}
          providers={[]}
          theme={theme}
          redirectTo={window.location.origin + "/app/"}
        />
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
