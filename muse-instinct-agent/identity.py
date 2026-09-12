"""Caller identity for the deployment.

Default: LangSmith API key. Callers must present a valid LangSmith workspace
API key (sent as `x-api-key`); LangSmith Cloud verifies it. This locks the
deployment to key holders and is what `mda dev` / SDK clients use.

For signed-in end users with PRIVATE threads (so caller A's Link session and
threads are not visible to caller B), switch to Supabase:

    identity = define_identity(auth=auth.supabase(project_ref="your-project-ref"))

That is the prerequisite for genuine per-user payments — each caller would then
authorize their own Link wallet. It needs a Supabase project and a frontend
that forwards the user's access token, so it is left as a documented follow-up.
"""

from __future__ import annotations

from managed_deepagents import auth, define_identity

identity = define_identity(auth=auth.langsmith_api_key())
