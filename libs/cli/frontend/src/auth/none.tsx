import type { ReactNode } from "react";

import type { AuthAdapter, SessionState } from "./types";

const session: SessionState = {
  status: "signed-in",
  accessToken: "",
  userIdentity: "anonymous",
  userEmail: null,
  signOut: async () => {},
};

const adapter: AuthAdapter = {
  Provider: ({ children }: { children: ReactNode }) => <>{children}</>,
  useSession: () => session,
  AuthUI: () => null,
};

export default adapter;
