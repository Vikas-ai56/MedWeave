"""`python -m tracking ui [--port 5000]`: open the MLflow UI on this machine's MedWeave tracking store."""

import argparse
import os
import sys

from tracking.locations import tracking_dir, tracking_uri


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m tracking")
    parser.add_argument("command", choices=["ui"])
    parser.add_argument("--port", type=int, default=5000)
    options = parser.parse_args()
    tracking_dir().mkdir(parents=True, exist_ok=True)
    print(f"MLflow UI for {tracking_uri()} at http://127.0.0.1:{options.port}", flush=True)
    arguments = [sys.executable, "-m", "mlflow", "ui", "--backend-store-uri", tracking_uri(), "--host", "127.0.0.1", "--port", str(options.port)]
    os.execv(sys.executable, arguments)
    return 0


if __name__ == "__main__":
    sys.exit(main())
