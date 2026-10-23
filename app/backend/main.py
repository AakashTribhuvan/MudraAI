"""MUDRA backend entry point.

    python main.py            (from app/backend, with the .venv active; port 8000, or set MUDRA_PORT)
    uvicorn main:app --reload
"""

import logging
import os

from mudra_api.app import create_app
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=int(os.getenv("MUDRA_PORT", "8000")))
