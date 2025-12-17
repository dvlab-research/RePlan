import asyncio
import base64
import logging
import os
import traceback
from io import BytesIO
from typing import Optional, Dict

import aiohttp
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel

from replan.pipelines.replan import RePlanPipeline
from PIL import Image


logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("replan.kontext_server")


class RegionEditInput(BaseModel):
    image_base64: str
    instruction: str
    response: str
    delete_main_prompt: bool = False
    replace_global_prompt: bool = False
    custom_global_prompt_text: str = "keep remaining parts of this image unchanged"
    expand_value: float = 0.15
    expand_mode: str = "ratio"
    attention_rules: Optional[Dict[str, bool]] = None
    bboxes_attend_to_each_other: bool = True


class HttpClient:
    session: aiohttp.ClientSession = None

    def start(self):
        self.session = aiohttp.ClientSession()

    async def stop(self):
        await self.session.close()
        self.session = None

    def __call__(self) -> aiohttp.ClientSession:
        assert self.session is not None
        return self.session


class RegionEditService:
    evaluator: RePlanPipeline = None
    device: str
    lock: asyncio.Lock

    def __init__(self, device: str):
        self.device = device
        self.lock = asyncio.Lock()

    def start(self):
        flux_model_name = "black-forest-labs/FLUX.1-Kontext-dev"
        logger.info(f"Loading RePlanPipeline (flux) on device='{self.device}' ...")
        # CPU inference typically does not support bfloat16 efficiently; default to float32.
        torch_dtype = torch.bfloat16 if self.device != "cpu" else torch.float32
        self.evaluator = RePlanPipeline(
            pipeline_type="flux",
            diffusion_model_name=flux_model_name,
            device=self.device,
            torch_dtype=torch_dtype,
            init_vlm=False
        )

    def __call__(self) -> RePlanPipeline:
        assert self.evaluator is not None
        return self.evaluator


app = FastAPI()
service_url = os.getenv("SERVICE_URL", "http://localhost:18000")
http_client = HttpClient()

# Configure CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods, e.g., GET, POST, OPTIONS, etc.
    allow_headers=["*"],  # Allows all headers
)


async def load_model():
    try:
        logger.info("Model loading started in background. Please wait until status becomes READY.")
        device = None
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            logger.info(f"Detected CUDA with {num_gpus} visible GPU(s). Using device cuda:0.")
            if num_gpus > 1:
                logger.warning(
                    f"Worker process sees {num_gpus} GPUs, but will only use cuda:0. "
                    "Consider setting CUDA_VISIBLE_DEVICES to assign a specific GPU for this worker."
                )
            device = "cuda:0"
        elif torch.backends.mps.is_available():
            logger.info("Detected Apple MPS. Using device mps.")
            device = "mps"
        else:
            # Allow CPU fallback to avoid 'server up but never loads' behavior in CPU-only environments.
            logger.warning("No CUDA/MPS detected. Falling back to CPU (this will be slow).")
            device = "cpu"

        pipeline = RegionEditService(device=device)
        await asyncio.to_thread(pipeline.start)
        
        app.state.pipeline = pipeline
        app.state.model_ready = True
        logger.info("READY: model loaded successfully and service can accept requests.")

    except Exception as e:
        logger.error(f"Model loading failed: {e}")
        logger.error(traceback.format_exc())
        app.state.model_load_failed = True


@app.on_event("startup")
async def startup():
    http_client.start()
    app.state.model_ready = False
    app.state.model_load_failed = False
    logger.info("Server started. Model is NOT ready yet; it will be loaded asynchronously in background.")
    logger.info("Scheduling background model load task ...")
    asyncio.create_task(load_model())


def save_image(image):
    # No longer needed
    pass


@app.get("/")
@app.post("/")
@app.options("/")
async def base():
    return "Welcome to Diffusers! Where you can use diffusion models to generate images"

@app.get("/health")
async def health():
    """
    Lightweight readiness probe for orchestration / startup scripts.
    status: 'loading' | 'ready' | 'failed'
    """
    failed = bool(getattr(app.state, "model_load_failed", False))
    ready = bool(getattr(app.state, "model_ready", False)) and hasattr(app.state, "pipeline")
    if failed:
        status = "failed"
    elif ready:
        status = "ready"
    else:
        status = "loading"
    return {
        "status": status,
        "model_ready": ready,
        "model_load_failed": failed,
    }

@app.post("/v1/images/generations")
async def generate_image(edit_input: RegionEditInput):
    try:
        if getattr(app.state, 'model_load_failed', False):
            raise HTTPException(status_code=500, detail="Model loading failed permanently. Check server logs.")
        
        if not getattr(app.state, 'model_ready', False) or not hasattr(app.state, "pipeline"):
            raise HTTPException(status_code=503, detail="Service not ready. Model is still loading. Please try again later.")
        
        selected_pipeline = app.state.pipeline
        
        async with selected_pipeline.lock:
            evaluator = selected_pipeline()

            image = None
            if edit_input.image_base64:
                try:
                    image_data = base64.b64decode(edit_input.image_base64)
                    image = Image.open(BytesIO(image_data))
                except Exception as e:
                    raise HTTPException(status_code=400, detail=f"Invalid base64 image data: {e}")
            
            if image is None:
                 raise HTTPException(status_code=400, detail="image_base64 must be provided.")

            parsed_attention_rules = None
            if edit_input.attention_rules:
                parsed_attention_rules = {}
                for key, value in edit_input.attention_rules.items():
                    try:
                        parts = [part.strip() for part in key.split(",")]
                        if len(parts) != 2 or not all(parts):
                            raise ValueError("Key must have two non-empty parts separated by a comma.")
                        q, k = parts
                        parsed_attention_rules[(q, k)] = value
                    except ValueError:
                        raise HTTPException(
                            status_code=400,
                            detail=f"Invalid attention_rules key format: '{key}'. Expected 'query, key'.",
                        )

            def edit_fn():
                # Directly use region_edit_with_attention and skip disk I/O
                edited_image, _, _ = evaluator.region_edit_with_attention(
                    image=image,
                    instruction=edit_input.instruction,
                    response=edit_input.response,
                    delete_main_prompt=edit_input.delete_main_prompt,
                    replace_global_prompt=edit_input.replace_global_prompt,
                    custom_global_prompt_text=edit_input.custom_global_prompt_text,
                    expand_value=edit_input.expand_value,
                    expand_mode=edit_input.expand_mode,
                    attention_rules=parsed_attention_rules,
                    bboxes_attend_to_each_other=edit_input.bboxes_attend_to_each_other,
                    skip_save=True
                )

                # Encode the edited image to base64
                buffered = BytesIO()
                edited_image.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

                return {"final_image_base64": img_str}

            logger.info("Request accepted: running image generation in a worker thread ...")
            results = await asyncio.to_thread(edit_fn)
            logger.info("Request completed: image generation finished.")
            json_compatible_results = jsonable_encoder(results)
            return {"data": [json_compatible_results]}
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        # Log the full traceback on the server for debugging
        logger.error(f"Unhandled error in generate_image: {e}")
        logger.error(traceback.format_exc())
        # Raise a clean HTTPException for the client
        raise HTTPException(status_code=500, detail="Internal Server Error. Check server logs for details.")


# if __name__ == "__main__":
#     import uvicorn

#     uvicorn.run(app, host="0.0.0.0", port=18000)