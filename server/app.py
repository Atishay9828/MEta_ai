"""
Server entry point for OpenEnv.
Hosts the negotiation environment as a FastAPI server.
"""
import sys
import os

# Add parent directory to path so we can import from root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app  # noqa: E402

__all__ = ["app"]


def main():
    """Run the server with uvicorn."""
    import uvicorn
    port = int(os.environ.get("PORT", "7860"))
    host = os.environ.get("HOST", "0.0.0.0")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
