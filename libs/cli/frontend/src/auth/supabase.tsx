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

  return (
    <SupabaseCtx.Provider value={{ supabase, state }}>
      {children}
    </SupabaseCtx.Provider>
  );
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
  const [mode, setMode] = useState<"signin" | "signup">("signin");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [confirmation, setConfirmation] = useState<string | null>(null);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setConfirmation(null);
    try {
      if (mode === "signup") {
        const { error } = await supabase.auth.signUp({ email, password });
        if (error) throw error;
        setConfirmation("Check your inbox to confirm your email.");
      } else {
        const { error } = await supabase.auth.signInWithPassword({
          email,
          password,
        });
        if (error) throw error;
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-dvh flex items-center justify-center bg-[var(--background)] p-4">
      <form
        onSubmit={submit}
        className="flex flex-col gap-3 w-full max-w-sm rounded-xl border border-[var(--border)] bg-[var(--surface)] p-6 shadow-sm"
      >
        <div className="flex items-center gap-2">
          <img
            src={theme === "dark" ? "/app/logo-dark.svg" : "/app/logo-light.svg"}
            alt="Deep Agents"
            className="h-8 w-8 rounded"
          />
          <h1 className="text-xl font-semibold text-[var(--foreground)]">
            {mode === "signin" ? "Sign in" : "Sign up"}
          </h1>
        </div>
        <input
          type="email"
          required
          placeholder="you@example.com"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          className="rounded-md border border-[var(--input-border)] bg-[var(--surface)] px-3 py-2 text-sm text-[var(--foreground)] placeholder:text-[var(--muted-foreground)] focus:border-[var(--input-focus)] focus:outline-none focus:ring-2 focus:ring-[var(--accent-bg)]"
        />
        <input
          type="password"
          required
          minLength={6}
          placeholder="password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          className="rounded-md border border-[var(--input-border)] bg-[var(--surface)] px-3 py-2 text-sm text-[var(--foreground)] placeholder:text-[var(--muted-foreground)] focus:border-[var(--input-focus)] focus:outline-none focus:ring-2 focus:ring-[var(--accent-bg)]"
        />
        <button
          type="submit"
          disabled={loading}
          className="rounded-md bg-[var(--primary)] px-3 py-2 text-sm font-medium text-[var(--primary-foreground)] transition-colors hover:opacity-90 disabled:opacity-50"
        >
          {loading ? "…" : mode === "signin" ? "Sign in" : "Sign up"}
        </button>
        {error && <p className="text-xs text-red-500">{error}</p>}
        {confirmation && <p className="text-xs text-emerald-500">{confirmation}</p>}
        <button
          type="button"
          className="text-center text-xs text-[var(--muted-foreground)] hover:underline"
          onClick={() => {
            setMode(mode === "signin" ? "signup" : "signin");
            setError(null);
            setConfirmation(null);
          }}
        >
          {mode === "signin"
            ? "Need an account? Sign up"
            : "Already have an account? Sign in"}
        </button>
      </form>
    </div>
  );
}

const adapter: AuthAdapter = {
  Provider: SupabaseProvider,
  useSession,
  AuthUI: SupabaseAuthUI,
};

export default adapter;
