import asyncio
import logging
import os
from typing import List, Dict

import aiohttp
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import Response
import uvicorn

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("replan.load_balancer")

# --- Configuration ---
# Read worker service URLs from env
worker_urls_str = os.getenv("WORKER_URLS", "http://localhost:18000,http://localhost:18001")
WORKER_URLS: List[str] = [url.strip() for url in worker_urls_str.split(',') if url.strip()]

# Load balancer listen port
LOAD_BALANCER_PORT = int(os.getenv("PORT", "8001"))
# --- End Configuration ---

app = FastAPI()

# track active connections for each worker
worker_stats: Dict[str, int] = {url: 0 for url in WORKER_URLS}
lock = asyncio.Lock()

client_session: aiohttp.ClientSession

@app.on_event("startup")
async def startup_event():
    global client_session
    client_session = aiohttp.ClientSession()
    logger.info(f"Load balancer starting on 0.0.0.0:{LOAD_BALANCER_PORT}")
    logger.info(f"Configured workers: {WORKER_URLS}")

@app.on_event("shutdown")
async def shutdown_event():
    await client_session.close()

async def forward_request(request: Request, target_url: str):
    data = await request.body()
    headers = {key: value for key, value in request.headers.items() if key.lower() not in ['host']}

    try:
        async with client_session.request(
            method=request.method,
            url=target_url,
            data=data,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=300) 
        ) as response:
            response_body = await response.read()
            
            return Response(
                content=response_body,
                status_code=response.status,
                headers=response.headers
            )
    except aiohttp.ClientConnectorError:
        logger.error(f"Cannot connect to worker at {target_url}")
        raise HTTPException(status_code=503, detail=f"Service unavailable: cannot connect to worker.")
    except asyncio.TimeoutError:
        logger.error(f"Request to worker at {target_url} timed out.")
        raise HTTPException(status_code=504, detail=f"Gateway timeout: worker did not respond in time.")

@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD", "PATCH"])
async def reverse_proxy(request: Request, path: str):
    """
    reverse proxy main entry.
    use "least connections" strategy to select worker and forward request.
    """
    selected_worker_url = None
    
    # --- core logic: select the least busy worker ---
    async with lock:
        # find the worker with the least active connections
        selected_worker_url = min(worker_stats, key=worker_stats.get)
        # increase the connection count for the selected worker
        worker_stats[selected_worker_url] += 1
        active_connections = worker_stats[selected_worker_url]
    
    logger.info(f"Proxy {request.method} /{path} -> {selected_worker_url} (active={active_connections})")

    try:
        # build the complete worker service URL
        target_url = f"{selected_worker_url}/{path}"
        return await forward_request(request, target_url)
    finally:
        # --- core logic: decrease the connection count after the request ---
        if selected_worker_url:
            async with lock:
                worker_stats[selected_worker_url] -= 1
            # Intentionally keep completion log at DEBUG to reduce noise under load.
            logger.debug(f"Finished -> {selected_worker_url} (active={worker_stats[selected_worker_url]})")


@app.get("/health")
async def health():
    """Simple health endpoint for the load balancer."""
    async with lock:
        stats = dict(worker_stats)
    return {
        "status": "ok",
        "workers": WORKER_URLS,
        "active_connections": stats,
    }


if __name__ == "__main__":
    if not WORKER_URLS:
        logger.error("WORKER_URLS is empty. Set env WORKER_URLS to a comma-separated list of worker URLs.")
        exit(1)
        
    uvicorn.run(app, host="0.0.0.0", port=LOAD_BALANCER_PORT) 