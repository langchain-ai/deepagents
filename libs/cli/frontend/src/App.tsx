import { useAuthAdapter } from "./auth/loader";
import type { AuthAdapter } from "./auth/types";
import { ASSISTANT_ID } from "./constants";
import RuntimeProvider from "./RuntimeProvider";
import Thread from "./components/Thread";
import AppHeader from "./components/AppHeader";

export default function App() {
  const adapter = useAuthAdapter();
  if (!adapter) return <SplashScreen />;
  return (
    <adapter.Provider>
      <Gate adapter={adapter} />
    </adapter.Provider>
  );
}

function Gate({ adapter }: { adapter: AuthAdapter }) {
  const session = adapter.useSession();
  if (session.status === "loading") return <SplashScreen />;
  if (session.status === "signed-out") {
    const { AuthUI } = adapter;
    return <AuthUI />;
  }
  return (
    <AuthenticatedApp
      accessToken={session.accessToken}
      userEmail={session.userEmail}
      onSignOut={session.signOut}
    />
  );
}

function SplashScreen() {
  return <div className="min-h-dvh bg-slate-50" />;
}

function AuthenticatedApp({
  accessToken,
  userEmail,
  onSignOut,
}: {
  accessToken: string;
  userEmail: string | null;
  onSignOut: () => Promise<void>;
}) {
  return (
    <RuntimeProvider accessToken={accessToken} assistantId={ASSISTANT_ID}>
      <div className="flex h-dvh flex-col bg-[var(--background)]">
        <AppHeader userEmail={userEmail} onSignOut={onSignOut} />
        <Thread />
      </div>
    </RuntimeProvider>
  );
}
