import type { ReactNode } from "react";
import { APP_DESCRIPTION, APP_NAME } from "../constants";
import { useTheme } from "../ThemeProvider";

export default function AppHeader({
  userEmail,
  isAnonymous,
  onSignOut,
  threadPicker,
}: {
  userEmail: string | null;
  isAnonymous: boolean;
  onSignOut: () => Promise<void>;
  threadPicker: ReactNode;
}) {
  const { theme, toggleTheme } = useTheme();
  const logoSrc = theme === "dark" ? "/app/logo-dark.svg" : "/app/logo-light.svg";

  return (
    <header className="header-blur sticky top-0 z-30 flex flex-wrap items-center justify-between gap-2 border-b border-[var(--border)] px-3 py-2 sm:px-6 sm:py-3">
      <div className="flex items-center gap-2 sm:gap-3">
        <img
          src={logoSrc}
          alt="Deep Agents"
          className="h-7 w-7 rounded sm:h-8 sm:w-8"
          width={32}
          height={32}
        />
        <div>
          <h1 className="text-sm font-semibold sm:text-lg">{APP_NAME}</h1>
          <p className="hidden text-xs text-[var(--muted-foreground)] sm:block">{APP_DESCRIPTION}</p>
        </div>
      </div>
      <div className="flex items-center gap-1.5 sm:gap-2">
        {userEmail && (
          <span className="hidden max-w-[160px] truncate text-xs text-[var(--muted-foreground)] md:inline">
            {userEmail}
          </span>
        )}
        <button
          type="button"
          onClick={toggleTheme}
          aria-label={theme === "dark" ? "Switch to light mode" : "Switch to dark mode"}
          title={theme === "dark" ? "Switch to light mode" : "Switch to dark mode"}
          className="flex h-7 w-7 items-center justify-center rounded-lg border border-[var(--border)] text-[var(--muted-foreground)] transition-colors hover:bg-[var(--accent-bg)]"
        >
          {theme === "dark" ? (
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" className="h-3.5 w-3.5">
              <circle cx="12" cy="12" r="4" />
              <path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M4.93 19.07l1.41-1.41M17.66 6.34l1.41-1.41" />
            </svg>
          ) : (
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" className="h-3.5 w-3.5">
              <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" />
            </svg>
          )}
        </button>
        {!isAnonymous && (
          <button
            onClick={() => { void onSignOut(); }}
            className="rounded-lg border border-[var(--border)] px-3 py-1.5 text-xs font-medium text-[var(--muted-foreground)] transition-colors hover:bg-[var(--accent-bg)]"
          >
            Sign out
          </button>
        )}
        {threadPicker}
      </div>
    </header>
  );
}
