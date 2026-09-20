# Vercel serverless entrypoint - exposes FastAPI app as `app`
# Vercel expects `app` variable in api/index.py
from app.main import app  # noqa: F401
