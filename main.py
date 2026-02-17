"""
Entry point for Railway deployment.
Imports and exposes the FastAPI app from research_system.api.main
"""
from research_system.api.main import app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
