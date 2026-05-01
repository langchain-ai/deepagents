import type { ReactNode, ComponentType } from "react";

export type SessionState =
  | { status: "loading" }
  | { status: "signed-out" }
  | {
      status: "signed-in";
      accessToken: string;
      userIdentity: string;
      userEmail: string | null;
      isAnonymous: boolean;
      signOut: () => Promise<void>;
    };

export interface AuthAdapter {
  /** Provider component that initializes the auth SDK. Wraps the tree. */
  Provider: ComponentType<{ children: ReactNode }>;
  /** Returns the current session state. Must be called inside `Provider`. */
  useSession: () => SessionState;
  /** Sign-in / sign-up UI rendered when session is "signed-out". */
  AuthUI: ComponentType;
}
