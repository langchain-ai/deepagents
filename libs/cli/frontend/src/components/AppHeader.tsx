import { APP_DESCRIPTION, APP_NAME } from "../constants";
import ThreadPicker from "./ThreadPicker";

export default function AppHeader({
  userEmail,
  onSignOut,
}: {
  userEmail: string | null;
  onSignOut: () => Promise<void>;
}) {
  return (
    <header className="header-blur sticky top-0 z-30 flex flex-wrap items-center justify-between gap-2 border-b border-[var(--border)] px-3 py-2 sm:px-6 sm:py-3">
      <div className="flex items-center gap-2 sm:gap-3">
        <svg className="h-5 w-5 sm:h-6 sm:w-6" viewBox="0 0 24 24" fill="none" stroke="var(--accent)" strokeWidth="1.5">
          <path d="M12 2L2 12l10 10 10-10L12 2z" />
          <path d="M12 8L8 12l4 4 4-4-4-4z" />
        </svg>
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
          onClick={() => { void onSignOut(); }}
          className="rounded-lg border border-[var(--border)] px-3 py-1.5 text-xs font-medium text-[var(--muted-foreground)] transition-colors hover:bg-[var(--accent-bg)]"
        >
          Sign out
        </button>
        <ThreadPicker />
      </div>
    </header>
  );
}
