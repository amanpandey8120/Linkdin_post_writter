import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from generate import generate_post
from model_loader import load_model_and_tokenizer
import config

# ── Logging ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s | %(name)-18s | %(levelname)-5s | %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("app")

# Global flag so we know whether the model is ready
model_ready = False


# ── Lifespan (load model on startup) ─────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_ready
    logger.info("Starting up – loading model…")
    try:
        load_model_and_tokenizer()
        model_ready = True
        logger.info("Model loaded successfully on startup")
    except Exception as exc:
        logger.critical("Failed to load model on startup: %s", exc)
        logger.critical(
            "The server will still start, but /generate will return an error. "
            "Make sure you have enough RAM (~32GB for CPU) or a GPU with 6GB+ VRAM."
        )
        model_ready = False
    yield
    logger.info("Shutting down")


# ── FastAPI application ──────────────────────────────────────────────────
app = FastAPI(
    title="LinkedIn Post Writer AI",
    description="Generate professional LinkedIn posts using a fine-tuned Qwen3 8B model",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")


# ── Request / Response models ────────────────────────────────────────────
class PostRequest(BaseModel):
    topic: str = Field(..., min_length=1, max_length=200,
                       description="Main topic of the LinkedIn post")
    tone: str = Field(default="Professional",
                      description="Writing tone")
    industry: str = Field(default="Technology",
                          description="Industry context")
    audience: str = Field(default="General",
                          description="Target audience")
    post_length: str = Field(default="Medium",
                             description="Desired post length")
    cta: str = Field(default="Engage with the post",
                     description="Call to action")


class PostResponse(BaseModel):
    linkedin_post: str
    word_count: int
    char_count: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


# ── Routes ───────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def serve_index():
    with open("templates/index.html", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(status="ok" if model_ready else "error",
                          model_loaded=model_ready)


@app.post("/generate", response_model=PostResponse)
async def generate_linkedin_post(request: PostRequest):
    if not model_ready:
        raise HTTPException(
            status_code=503,
            detail=(
                "Model is not loaded. The server needs more memory to run this model. "
                "Minimum requirements: 6GB+ GPU VRAM or 32GB+ CPU RAM. "
                "Alternatively, deploy on a cloud GPU instance."
            ),
        )

    logger.info("Generate request: topic='%s', tone='%s', industry='%s'",
                request.topic, request.tone, request.industry)
    try:
        post = generate_post(
            topic=request.topic,
            tone=request.tone,
            industry=request.industry,
            audience=request.audience,
            post_length=request.post_length,
            cta=request.cta,
        )
        if not post or len(post.strip()) < 10:
            raise HTTPException(
                status_code=500,
                detail="Generated post is too short or empty. Please try again.",
            )
        return PostResponse(
            linkedin_post=post,
            word_count=len(post.split()),
            char_count=len(post),
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Generation failed")
        raise HTTPException(
            status_code=500,
            detail=f"Post generation failed: {str(exc)}",
        ) from exc


# ── Entry point ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host=config.HOST,
        port=config.PORT,
        reload=False,
        log_level=config.LOG_LEVEL.lower(),
    )
