-- 1. XÓA CÁC FUNCTION CŨ
DROP FUNCTION IF EXISTS public.search_destinations(vector, int, uuid[]);
DROP FUNCTION IF EXISTS public.search_trips(vector, int, uuid[]);
DROP FUNCTION IF EXISTS public.search_routes(vector, int, uuid[]);

-- 2. TẠO FUNCTION HYBRID CHO DESTINATIONS
CREATE OR REPLACE FUNCTION public.search_destinations(
  query_text text,
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
LANGUAGE sql STABLE SECURITY DEFINER SET search_path = public
AS $$
  SELECT
    d.id,
    -- Hybrid Score = Semantic Score + Keyword Score (0.5 nếu tên chứa query_text)
    (
      (1 - (d.embedding <=> query_embedding)) + 
      (CASE WHEN query_text <> '' AND d.name ILIKE '%' || query_text || '%' THEN 0.5 ELSE 0.0 END)
    )::double precision AS similarity,
    d.name,
    d.country,
    d.description,
    d.category,
    d.region_id
  FROM public.destinations d
  WHERE d.embedding IS NOT NULL
    AND (exclude_ids IS NULL OR cardinality(exclude_ids) = 0 OR NOT (d.id = ANY (exclude_ids)))
  ORDER BY similarity DESC
  LIMIT LEAST(coalesce(match_count, 5), 100);
$$;

-- 3. TẠO FUNCTION HYBRID CHO TRIPS
CREATE OR REPLACE FUNCTION public.search_trips(
  query_text text,
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
LANGUAGE sql STABLE SECURITY DEFINER SET search_path = public
AS $$
  SELECT
    t.id,
    (
      (1 - (t.embedding <=> query_embedding)) + 
      (CASE WHEN query_text <> '' AND t.title ILIKE '%' || query_text || '%' THEN 0.5 ELSE 0.0 END)
    )::double precision AS similarity,
    t.title,
    t.description,
    t.destination_id,
    t.departure,
    t.status
  FROM public.trips t
  WHERE t.embedding IS NOT NULL
    AND (exclude_ids IS NULL OR cardinality(exclude_ids) = 0 OR NOT (t.id = ANY (exclude_ids)))
  ORDER BY similarity DESC
  LIMIT LEAST(coalesce(match_count, 5), 100);
$$;

-- 4. TẠO FUNCTION HYBRID CHO ROUTES
CREATE OR REPLACE FUNCTION public.search_routes(
  query_text text,
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
LANGUAGE sql STABLE SECURITY DEFINER SET search_path = public
AS $$
  SELECT
    r.id,
    (
      (1 - (r.embedding <=> query_embedding)) + 
      (CASE WHEN query_text <> '' AND r.title ILIKE '%' || query_text || '%' THEN 0.5 ELSE 0.0 END)
    )::double precision AS similarity,
    r.title,
    r.description,
    r.trip_id,
    r."index" AS route_index
  FROM public.routes r
  WHERE r.embedding IS NOT NULL
    AND (exclude_ids IS NULL OR cardinality(exclude_ids) = 0 OR NOT (r.id = ANY (exclude_ids)))
  ORDER BY similarity DESC
  LIMIT LEAST(coalesce(match_count, 5), 100);
$$;

-- CẤP QUYỀN
GRANT EXECUTE ON FUNCTION public.search_destinations(text, vector, int, uuid[]) TO anon, authenticated, service_role;
GRANT EXECUTE ON FUNCTION public.search_trips(text, vector, int, uuid[]) TO anon, authenticated, service_role;
GRANT EXECUTE ON FUNCTION public.search_routes(text, vector, int, uuid[]) TO anon, authenticated, service_role;