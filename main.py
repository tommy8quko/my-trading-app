"""Entry point — starts FastAPI backend then opens pywebview window."""
import threading
import time
import uvicorn
import webview

PORT = 7432


def _start_server():
    from backend.api import app
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")


if __name__ == "__main__":
    # Start FastAPI in background thread
    t = threading.Thread(target=_start_server, daemon=True)
    t.start()

    # Wait briefly for server to be ready
    time.sleep(1.5)

    # Open desktop window
    webview.create_window(
        "Trading Journal",
        url=f"http://127.0.0.1:{PORT}",
        width=1400,
        height=860,
        min_size=(1024, 600),
    )
    webview.start()
