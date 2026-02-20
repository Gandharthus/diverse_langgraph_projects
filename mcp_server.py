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
            try:
                return e.response.json()
            except Exception:
                return {"error": str(e)}
        except Exception as e:
            return {"error": str(e)}


@mcp.tool()
async def resolve_index(pattern: str, top_k: int = 5) -> dict | list[dict]:
    """
    Resolve Elasticsearch indices matching a pattern.
    """
    url = f"{ES_URL}/_cat/indices/{pattern}?format=json"
    
    async with httpx.AsyncClient(verify=VERIFY_SSL) as client:
        try:
            response = await client.get(
                url,
                auth=ES_AUTH,
                timeout=10.0,
            )
            response.raise_for_status()
            data = response.json()
            
            indices: list[dict] = []
            for item in data:
                name = item.get("index")
                if not name:
                    continue
                try:
                    docs_count = int(item.get("docs.count", 0))
                except (TypeError, ValueError):
                    docs_count = 0
                index_info = {
                    "name": name,
                    "health": item.get("health"),
                    "status": item.get("status"),
                    "docs_count": docs_count,
                    "store_size": item.get("store.size"),
                }
                indices.append(index_info)
            
            indices.sort(key=lambda x: (-x["docs_count"], x["name"]))
            return indices[:top_k]
        except httpx.HTTPStatusError as e:
            try:
                return e.response.json()
            except Exception:
                return {"error": str(e)}
        except Exception as e:
            return {"error": str(e)}


if __name__ == "__main__":
    mcp.run(transport="stdio")
