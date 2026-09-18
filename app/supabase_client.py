import os

from supabase import Client, create_client


SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_ANON_KEY = os.environ.get("SUPABASE_ANON_KEY")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get(
    "SUPABASE_SERVICE_ROLE_KEY"
)

missing = [
    name
    for name, value in {
        "SUPABASE_URL": SUPABASE_URL,
        "SUPABASE_ANON_KEY": SUPABASE_ANON_KEY,
        "SUPABASE_SERVICE_ROLE_KEY": SUPABASE_SERVICE_ROLE_KEY,
    }.items()
    if not value
]

if missing:
    raise RuntimeError(
        "Missing required environment variables: "
        + ", ".join(missing)
    )


# Used only to validate a user access token:
# supabase_auth.auth.get_user(access_token)
supabase_auth: Client = create_client(
    SUPABASE_URL,
    SUPABASE_ANON_KEY,
)


# Backend-only client. The service-role key bypasses RLS.
# Never expose it in frontend code, notebooks, screenshots, logs,
# Git history, or a public repository.
supabase_admin: Client = create_client(
    SUPABASE_URL,
    SUPABASE_SERVICE_ROLE_KEY,
)
