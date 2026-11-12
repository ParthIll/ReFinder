"""Entry point for the packaged backend (PyInstaller) that the desktop app starts and stops itself.

Runs the same FastAPI app as `uvicorn evaluation.server:app`. The app passes REFINDER_DATA_DIR (where the index
lives), REFINDER_PARENT_PID (quit when the app quits) and optionally REFINDER_PORT.
"""

import multiprocessing
import os


def main() -> None:
    multiprocessing.freeze_support()  # extraction worker processes re-enter this executable
    import uvicorn

    from evaluation.server import app

    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("REFINDER_PORT", "8000")), log_level="info")


if __name__ == "__main__":
    main()
