from mcp.server.fastmcp import FastMCP
import httpx
import json

# Initialize FastMCP server
mcp = FastMCP("Elasticsearch Simple Server")

ES_URL = "https://localhost:9200"
ES_AUTH = ("elastic", "changeme")
VERIFY_SSL = False  # Set to False for self-signed certificates in dev

@mcp.tool()
async def healthcheck() -> bool:
    """
    Check if the MCP server is connected to Elasticsearch.
    Returns True if Elasticsearch responds with the tagline "You Know, for Search".
    """
    try:
        async with httpx.AsyncClient(verify=VERIFY_SSL) as client:
            response = await client.get(ES_URL, auth=ES_AUTH, timeout=5.0)
            if response.status_code == 200:
                data = response.json()
                if data.get("tagline") == "You Know, for Search":
                    return True
            return False
    except Exception:
        return False

@mcp.tool()
async def simulate_pipeline(pipeline: dict, docs: list[dict]) -> dict:
    """
    Simulate an ingest pipeline against a set of documents.
    Wraps the Elasticsearch /_ingest/pipeline/_simulate API.
    
    Args:
        pipeline: The pipeline definition (must contain 'processors' list).
        docs: A list of documents to simulate. Each document should be wrapped in 
              the standard _simulate structure (e.g. {'_source': {...}}).
    """
    url = f"{ES_URL}/_ingest/pipeline/_simulate"
    
    # Construct the body for the _simulate API
    # The API expects:
    # {
    #   "pipeline": { ... definition ... },
    #   "docs": [ ... ]
    # }
    body = {
        "pipeline": pipeline,
        "docs": docs
    }
    
    async with httpx.AsyncClient(verify=VERIFY_SSL) as client:
        try:
            response = await client.post(
                url, 
                json=body, 
                auth=ES_AUTH, 
                timeout=30.0
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            # If ES returns an error (e.g. 400 for bad pipeline), return the error details
            try:
                return e.response.json()
            except Exception:
                return {"error": str(e)}
        except Exception as e:
            return {"error": str(e)}

if __name__ == "__main__":
    mcp.run(transport="stdio")
