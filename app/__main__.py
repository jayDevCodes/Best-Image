"""Run Best-Image with: python -m app."""
import os
import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=os.getenv("BEST_IMAGE_HOST", "127.0.0.1"),
        port=int(os.getenv("BEST_IMAGE_PORT", "8765")),
        reload=False,
        log_level="info",
    )
