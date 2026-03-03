import os
import httpx
from fastapi import FastAPI, Request, HTTPException, Response
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI()

# Configuration
# The upstream API base URL (e.g. your LangGraph Cloud deployment URL)
# Ensure no trailing slash
UPSTREAM_API_BASE = os.getenv("LANGGRAPH_API_URL", "https://api.langchain.plus").rstrip("/")
# The API Key for the upstream service
UPSTREAM_API_KEY = os.getenv("LANGGRAPH_API_KEY", "")

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/threads")
async def create_thread(request: Request):
    """
    Proxy for creating a thread.
    """
    body = await request.body()
    url = f"{UPSTREAM_API_BASE}/threads"
    headers = {
        "x-api-key": UPSTREAM_API_KEY,
        "Content-Type": "application/json"
    }
    
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.post(url, content=body, headers=headers, timeout=10.0)
            # Forward the response status and body
            return Response(content=resp.content, status_code=resp.status_code, media_type=resp.headers.get("content-type"))
        except httpx.RequestError as e:
            raise HTTPException(status_code=502, detail=f"Upstream request failed: {str(e)}")

@app.post("/threads/{thread_id}/runs/stream")
async def stream_run(thread_id: str, request: Request):
    """
    Proxy for streaming a run.
    """
    body = await request.body()
    url = f"{UPSTREAM_API_BASE}/threads/{thread_id}/runs/stream"
    headers = {
        "x-api-key": UPSTREAM_API_KEY,
        "Content-Type": "application/json"
    }

    # We define a generator that yields chunks from the upstream stream
    async def event_generator():
        async with httpx.AsyncClient() as client:
            try:
                async with client.stream("POST", url, content=body, headers=headers, timeout=60.0) as response:
                    if response.status_code >= 400:
                        yield f"event: error\ndata: {{\"error\": \"Upstream error {response.status_code}\"}}\n\n".encode()
                        return

                    async for chunk in response.aiter_bytes():
                        yield chunk
            except httpx.RequestError as e:
                yield f"event: error\ndata: {{\"error\": \"Upstream connection failed: {str(e)}\"}}\n\n".encode()

    return StreamingResponse(event_generator(), media_type="text/event-stream")

# Serve the frontend
@app.get("/", response_class=HTMLResponse)
async def serve_index():
    # Read the file content
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        index_path = os.path.join(current_dir, "index.html")
        with open(index_path, "r", encoding="utf-8") as f:
            content = f.read()
        return content
    except FileNotFoundError:
        return "Frontend index.html not found", 404

if __name__ == "__main__":
    print(f"Starting proxy server...")
    print(f"Upstream API: {UPSTREAM_API_BASE}")
    # Check if API key is set
    if not UPSTREAM_API_KEY:
        print("WARNING: LANGGRAPH_API_KEY is not set. Upstream requests may fail.")
        
    port = int(os.getenv("PORT", 8081))
    uvicorn.run(app, host="0.0.0.0", port=port)
