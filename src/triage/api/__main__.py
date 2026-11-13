import argparse

import uvicorn

from triage.api.main import create_app

# Lancement local : TRIAGE_API_KEY=... python -m triage.api [--port 8000]
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run(create_app(), host=args.host, port=args.port)
