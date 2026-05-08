from __future__ import annotations
from supabase import create_client, Client
from config import get_secret

_client: Client | None = None


def get_supabase() -> Client:
    global _client
    if _client is None:
        url = get_secret("SUPABASE_URL")
        key = get_secret("SUPABASE_KEY")
        if not url or not key:
            raise RuntimeError("SUPABASE_URL and SUPABASE_KEY must be set in secrets or env vars.")
        _client = create_client(url, key)
    return _client
