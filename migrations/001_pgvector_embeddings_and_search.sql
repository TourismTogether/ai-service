-- Run in Supabase SQL editor or via migration pipeline.
-- Requires: pgvector enabled on project (Supabase: Database > Extensions > vector).

CREATE EXTENSION IF NOT EXISTS vector;

ALTER TABLE public.destinations
  ADD COLUMN IF NOT EXISTS embedding vector(768);

ALTER TABLE public.trips
  ADD COLUMN IF NOT EXISTS embedding vector(768);

ALTER TABLE public.routes
  ADD COLUMN IF NOT EXISTS embedding vector(768);

-- Optional: approximate index after backfilling embeddings (uncomment when populated).
-- CREATE INDEX IF NOT EXISTS destinations_embedding_ivfflat
--   ON public.destinations USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
-- CREATE INDEX IF NOT EXISTS trips_embedding_ivfflat
--   ON public.trips USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
-- CREATE INDEX IF NOT EXISTS routes_embedding_ivfflat
--   ON public.routes USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

CREATE OR REPLACE FUNCTION public.search_destinations(
  query_embedding vector(768),
  match_count int DEFAULT 5,
  exclude_ids uuid[] DEFAULT ARRAY[]::uuid[]
)
RETURNS TABLE (
  id uuid,
  similarity double precision,
  name text,
  country text,
  description text,
  category text,
  region_id uuid
)
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = public
AS $$
  SELECT
    d.id,
    (1 - (d.embedding <=> query_embedding))::double precision AS similarity,
    d.name,
    d.country,
    d.description,
    d.category,
    d.region_id
  FROM public.destinations d
  WHERE d.embedding IS NOT NULL
    AND (
      exclude_ids IS NULL
      OR cardinality(exclude_ids) = 0
      OR NOT (d.id = ANY (exclude_ids))
    )
  ORDER BY d.embedding <=> query_embedding
  LIMIT LEAST(coalesce(match_count, 5), 100);
$$;

CREATE OR REPLACE FUNCTION public.search_trips(
  query_embedding vector(768),
  match_count int DEFAULT 5,
  exclude_ids uuid[] DEFAULT ARRAY[]::uuid[]
)
RETURNS TABLE (
  id uuid,
  similarity double precision,
  title text,
  description text,
  destination_id uuid,
  departure text,
  status text
)
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = public
AS $$
  SELECT
    t.id,
    (1 - (t.embedding <=> query_embedding))::double precision AS similarity,
    t.title,
    t.description,
    t.destination_id,
    t.departure,
    t.status
  FROM public.trips t
  WHERE t.embedding IS NOT NULL
    AND (
      exclude_ids IS NULL
      OR cardinality(exclude_ids) = 0
      OR NOT (t.id = ANY (exclude_ids))
    )
  ORDER BY t.embedding <=> query_embedding
  LIMIT LEAST(coalesce(match_count, 5), 100);
$$;

CREATE OR REPLACE FUNCTION public.search_routes(
  query_embedding vector(768),
  match_count int DEFAULT 5,
  exclude_ids uuid[] DEFAULT ARRAY[]::uuid[]
)
RETURNS TABLE (
  id uuid,
  similarity double precision,
  title text,
  description text,
  trip_id uuid,
  route_index int
)
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = public
AS $$
  SELECT
    r.id,
    (1 - (r.embedding <=> query_embedding))::double precision AS similarity,
    r.title,
    r.description,
    r.trip_id,
    r."index" AS route_index
  FROM public.routes r
  WHERE r.embedding IS NOT NULL
    AND (
      exclude_ids IS NULL
      OR cardinality(exclude_ids) = 0
      OR NOT (r.id = ANY (exclude_ids))
    )
  ORDER BY r.embedding <=> query_embedding
  LIMIT LEAST(coalesce(match_count, 5), 100);
$$;

GRANT EXECUTE ON FUNCTION public.search_destinations(vector, int, uuid[]) TO anon, authenticated, service_role;
GRANT EXECUTE ON FUNCTION public.search_trips(vector, int, uuid[]) TO anon, authenticated, service_role;
GRANT EXECUTE ON FUNCTION public.search_routes(vector, int, uuid[]) TO anon, authenticated, service_role;
