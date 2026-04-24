from functools import lru_cache

import httpx
from supabase import Client, create_client, ClientOptions

from app.config import get_settings


@lru_cache
def get_supabase() -> Client:
    s = get_settings()
    # PostgREST/Supabase on HTTP/2 can raise httpx/httpcore RemoteProtocolError
    # under concurrent requests; HTTP/1.1 is more stable for many short-lived queries.
    http = httpx.Client(
        http2=False,
        timeout=httpx.Timeout(120.0, connect=30.0),
    )
    return create_client(
        s.supabase_url,
        s.supabase_key,
        options=ClientOptions(httpx_client=http),
    )
