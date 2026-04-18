from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.model.embedding_model import get_model
from app.api import recommend, search
from app.config import get_settings

@asynccontextmanager
async def lifespan(app: FastAPI):
    get_model() 
    yield

def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="Tourism AI",
        description="Semantic search and recommendations for destinations, trips, and routes.",
        version="1.0.0",
        lifespan=lifespan,
    )

    if settings.cors_origins.strip() == "*":
        origins = ["*"]
    else:
        origins = [
            o.strip()
            for o in settings.cors_origins.split(",")
            if o.strip()
        ]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(search.router, prefix="/search", tags=["search"])
    app.include_router(recommend.router, prefix="/recommend", tags=["recommend"])

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()
