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
  /**
   * True when Supabase fired a PASSWORD_RECOVERY event and the user must
   * set a new password before accessing the app. While true,
   * `state.status` is forced to "signed-out" so `Gate` keeps rendering
   * the AuthUI (which branches on this flag).
   */
  recoveryMode: boolean;
  clearRecoveryMode: () => void;
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
  const [recoveryMode, setRecoveryMode] = useState(false);

  useEffect(() => {
    let active = true;
    supabase.auth.getSession().then(({ data }) => {
      if (!active) return;
      setSession(data.session);
      setLoading(false);
    });
    const { data: { subscription } } = supabase.auth.onAuthStateChange(
      (event, nextSession) => {
        if (!active) return;
        setSession(nextSession);
        setLoading(false);
        if (event === "PASSWORD_RECOVERY") {
          // Supabase briefly creates a session for the recovery flow so
          // we can call updateUser({ password }). Keep the user on the
          // AuthUI (via recoveryMode) until they pick a new password.
          setRecoveryMode(true);
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
          signOut: async () => {
            await supabase.auth.signOut();
          },
        }
      : { status: "signed-out" };

  // When in recovery, pretend we're signed out so Gate renders AuthUI.
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

// ─── UI primitives ─────────────────────────────────────────────────────────

const INPUT_CLS =
  "rounded-md border border-[var(--input-border)] bg-[var(--surface)] px-3 py-2 text-sm text-[var(--foreground)] placeholder:text-[var(--muted-foreground)] focus:border-[var(--input-focus)] focus:outline-none focus:ring-2 focus:ring-[var(--accent-bg)]";

const PRIMARY_BTN_CLS =
  "rounded-md bg-[var(--primary)] px-3 py-2 text-sm font-medium text-[var(--primary-foreground)] transition-colors hover:opacity-90 disabled:opacity-50 flex items-center justify-center gap-2";

function Spinner() {
  return (
    <svg
      className="h-4 w-4 animate-spin"
      viewBox="0 0 24 24"
      fill="none"
      aria-label="Loading"
    >
      <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="2.5" opacity="0.25" />
      <path
        d="M12 2a10 10 0 0 1 10 10"
        stroke="currentColor"
        strokeWidth="2.5"
        strokeLinecap="round"
      />
    </svg>
  );
}

function PasswordInput({
  value,
  onChange,
  placeholder = "password",
  required = true,
  minLength = 6,
}: {
  value: string;
  onChange: (v: string) => void;
  placeholder?: string;
  required?: boolean;
  minLength?: number;
}) {
  const [show, setShow] = useState(false);
  return (
    <div className="relative">
      <input
        type={show ? "text" : "password"}
        required={required}
        minLength={minLength}
        placeholder={placeholder}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className={`${INPUT_CLS} w-full pr-10`}
      />
      <button
        type="button"
        onClick={() => setShow((v) => !v)}
        aria-label={show ? "Hide password" : "Show password"}
        title={show ? "Hide password" : "Show password"}
        className="absolute inset-y-0 right-0 flex items-center px-3 text-[var(--muted-foreground)] hover:text-[var(--foreground)]"
      >
        {show ? (
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" className="h-4 w-4">
            <path d="M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 10 8 10 8a13.16 13.16 0 0 1-1.67 2.68" />
            <path d="M6.61 6.61A13.526 13.526 0 0 0 2 12s3 8 10 8a9.74 9.74 0 0 0 5.39-1.61" />
            <line x1="2" x2="22" y1="2" y2="22" />
            <path d="M14.12 14.12a3 3 0 1 1-4.24-4.24" />
          </svg>
        ) : (
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" className="h-4 w-4">
            <path d="M2 12s3-8 10-8 10 8 10 8-3 8-10 8-10-8-10-8z" />
            <circle cx="12" cy="12" r="3" />
          </svg>
        )}
      </button>
    </div>
  );
}

// ─── Auth UI ───────────────────────────────────────────────────────────────

type Mode = "signin" | "signup" | "forgot" | "recovery";

function SupabaseAuthUI() {
  const { supabase, recoveryMode, clearRecoveryMode } = useSupabaseCtx();
  const { theme } = useTheme();
  const [mode, setMode] = useState<Mode>(recoveryMode ? "recovery" : "signin");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [confirmation, setConfirmation] = useState<string | null>(null);

  // Jump straight to the recovery form if Supabase fires PASSWORD_RECOVERY
  // while AuthUI is mounted.
  useEffect(() => {
    if (recoveryMode && mode !== "recovery") {
      setMode("recovery");
      setError(null);
      setConfirmation(null);
    }
  }, [recoveryMode, mode]);

  const resetMessages = () => {
    setError(null);
    setConfirmation(null);
  };

  const switchMode = (next: Mode) => {
    setMode(next);
    resetMessages();
  };

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    resetMessages();
    try {
      if (mode === "signup") {
        const { error } = await supabase.auth.signUp({ email, password });
        if (error) throw error;
        setConfirmation("Check your inbox to confirm your email.");
      } else if (mode === "signin") {
        const { error } = await supabase.auth.signInWithPassword({
          email,
          password,
        });
        if (error) throw error;
      } else if (mode === "forgot") {
        const { error } = await supabase.auth.resetPasswordForEmail(email, {
          redirectTo: window.location.origin + "/app/",
        });
        if (error) throw error;
        setConfirmation("Check your inbox for a password reset link.");
      } else if (mode === "recovery") {
        const { error } = await supabase.auth.updateUser({ password });
        if (error) throw error;
        // Password updated — Supabase keeps the session alive, so clearing
        // recoveryMode lets Gate flip to the authenticated app.
        clearRecoveryMode();
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  };

  const title =
    mode === "signin"
      ? "Sign in"
      : mode === "signup"
        ? "Sign up"
        : mode === "forgot"
          ? "Reset password"
          : "Set new password";

  const submitLabel =
    mode === "signin"
      ? "Sign in"
      : mode === "signup"
        ? "Sign up"
        : mode === "forgot"
          ? "Send reset link"
          : "Update password";

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
          <h1 className="text-xl font-semibold text-[var(--foreground)]">{title}</h1>
        </div>

        {mode === "recovery" ? (
          <p className="text-xs text-[var(--muted-foreground)]">
            Choose a new password for your account.
          </p>
        ) : mode === "forgot" ? (
          <p className="text-xs text-[var(--muted-foreground)]">
            Enter your email and we'll send you a link to reset your password.
          </p>
        ) : null}

        {mode !== "recovery" && (
          <input
            type="email"
            required
            placeholder="you@example.com"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            className={INPUT_CLS}
          />
        )}

        {mode !== "forgot" && (
          <PasswordInput
            value={password}
            onChange={setPassword}
            placeholder={mode === "recovery" ? "new password" : "password"}
          />
        )}

        <button type="submit" disabled={loading} className={PRIMARY_BTN_CLS}>
          {loading ? <Spinner /> : submitLabel}
        </button>

        {error && <p className="text-xs text-red-500">{error}</p>}
        {confirmation && <p className="text-xs text-emerald-500">{confirmation}</p>}

        {/* Footer nav links */}
        <div className="flex flex-col gap-1 pt-1">
          {mode === "signin" && (
            <>
              <button
                type="button"
                className="text-center text-xs text-[var(--muted-foreground)] hover:underline"
                onClick={() => switchMode("forgot")}
              >
                Forgot password?
              </button>
              <button
                type="button"
                className="text-center text-xs text-[var(--muted-foreground)] hover:underline"
                onClick={() => switchMode("signup")}
              >
                Need an account? Sign up
              </button>
            </>
          )}
          {mode === "signup" && (
            <button
              type="button"
              className="text-center text-xs text-[var(--muted-foreground)] hover:underline"
              onClick={() => switchMode("signin")}
            >
              Already have an account? Sign in
            </button>
          )}
          {mode === "forgot" && (
            <button
              type="button"
              className="text-center text-xs text-[var(--muted-foreground)] hover:underline"
              onClick={() => switchMode("signin")}
            >
              ← Back to sign in
            </button>
          )}
        </div>
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
