"""
Entry point for the Amukh Capital backend.

Run with:
    uvicorn backend.main:app --reload --port 8000
"""

from backend.app import create_app

app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
