"""FastAPI app: dashboard, upload, pipeline control, MJPEG stream, WebSocket metrics."""
from __future__ import annotations

import asyncio
import csv
import io
import re
import shutil
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .detector import PersonTracker
from .pipeline import VideoPipeline
from .state import SharedState

ROOT = Path(__file__).resolve().parent.parent
STATIC = ROOT / "static"
SAMPLES = ROOT / "samples"
UPLOADS = ROOT / "uploads"
VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}

app = FastAPI(title="Facility Monitor POC")
state = SharedState()
tracker: PersonTracker | None = None
pipeline: VideoPipeline | None = None
_lock = threading.Lock()
_shutting_down = threading.Event()


@app.on_event("startup")
def _load_model() -> None:
    global tracker
    tracker = PersonTracker()


@app.on_event("shutdown")
def _shutdown() -> None:
    # Long-lived MJPEG / WebSocket connections would otherwise keep uvicorn waiting forever on Ctrl+C.
    _shutting_down.set()
    with _lock:
        _stop_pipeline()


# ---------------------------------------------------------------- pages
@app.get("/", response_class=HTMLResponse)
def index() -> str:
    """Serve the dashboard with asset URLs versioned by mtime, so a plain reload never runs stale CSS/JS."""
    html = (STATIC / "index.html").read_text()
    for name in ("styles.css", "app.js"):
        v = int((STATIC / name).stat().st_mtime)
        html = html.replace(f"/static/{name}", f"/static/{name}?v={v}")
    return html


class NoCacheStatic(StaticFiles):
    async def get_response(self, path, scope):  # type: ignore[override]
        resp = await super().get_response(path, scope)
        resp.headers["Cache-Control"] = "no-cache"  # revalidate every time; ETag keeps it cheap
        return resp


app.mount("/static", NoCacheStatic(directory=STATIC), name="static")


# ---------------------------------------------------------------- sources
def _list_dir(d: Path, kind: str) -> list[dict]:
    d.mkdir(exist_ok=True)
    out = []
    for p in sorted(d.iterdir()):
        if p.suffix.lower() in VIDEO_EXT and p.stat().st_size > 0:
            out.append({"id": f"{kind}:{p.name}", "name": p.name, "kind": kind, "size_mb": round(p.stat().st_size / 1e6, 1)})
    return out


@app.get("/api/sources")
def sources() -> dict[str, Any]:
    return {"sources": [{"id": "webcam:0", "name": "Webcam (camera 0)", "kind": "webcam"}]
            + _list_dir(SAMPLES, "sample") + _list_dir(UPLOADS, "upload")}


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)) -> dict[str, Any]:
    name = Path(file.filename or "video.mp4").name
    if Path(name).suffix.lower() not in VIDEO_EXT:
        raise HTTPException(400, f"Unsupported file type: {name}")
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    dest = UPLOADS / f"{datetime.now():%Y%m%d-%H%M%S}_{safe}"
    UPLOADS.mkdir(exist_ok=True)
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    return {"id": f"upload:{dest.name}", "name": dest.name, "size_mb": round(dest.stat().st_size / 1e6, 1)}


# ---------------------------------------------------------------- control
class StartRequest(BaseModel):
    source: str
    loop: bool = True
    max_speed: bool = False


def _resolve(source_id: str) -> tuple[str | int, str]:
    kind, _, name = source_id.partition(":")
    if kind == "webcam":
        return int(name or 0), "Webcam"
    base = {"sample": SAMPLES, "upload": UPLOADS}.get(kind)
    if base is None:
        raise HTTPException(400, f"Unknown source kind: {kind}")
    path = (base / Path(name).name)
    if not path.is_file():
        raise HTTPException(404, f"File not found: {name}")
    return str(path), path.name


def _stop_pipeline() -> None:
    global pipeline
    if pipeline is not None:
        pipeline.stop()
        pipeline.join(timeout=5)
        pipeline = None


@app.post("/api/start")
def start(req: StartRequest) -> dict[str, Any]:
    global pipeline
    assert tracker is not None
    src, label = _resolve(req.source)
    with _lock:
        _stop_pipeline()
        pipeline = VideoPipeline(state, tracker, src, label, loop=req.loop, max_speed=req.max_speed)
        pipeline.start()
    # give the source a moment to open so a bad file reports an error immediately
    time.sleep(0.6)
    if pipeline.error:
        raise HTTPException(400, pipeline.error)
    return {"ok": True, "source": label}


class SeekRequest(BaseModel):
    time: float | None = None   # absolute video seconds
    delta: float | None = None  # relative seconds, e.g. -10 / +10


@app.post("/api/seek")
def seek(req: SeekRequest) -> dict[str, Any]:
    if req.time is None and req.delta is None:
        raise HTTPException(400, "Provide time or delta")
    with _lock:
        if pipeline is None or not pipeline.is_alive():
            raise HTTPException(409, "No source is running")
        try:
            target = pipeline.seek(req.time, req.delta)
        except ValueError as e:
            raise HTTPException(409, str(e)) from e
    return {"ok": True, "time": round(target, 1)}


@app.post("/api/stop")
def stop() -> dict[str, Any]:
    with _lock:
        _stop_pipeline()
    return {"ok": True}


# ---------------------------------------------------------------- data
@app.get("/api/state")
def get_state() -> dict[str, Any]:
    return {**state.snapshot(), "history": state.history_list(), "events": state.events_list()}


@app.get("/api/settings")
def get_settings() -> dict[str, Any]:
    return state.get_settings().model_dump()


@app.put("/api/settings")
def put_settings(patch: dict[str, Any]) -> dict[str, Any]:
    try:
        return state.update_settings(patch).model_dump()
    except Exception as e:  # noqa: BLE001
        raise HTTPException(422, str(e)) from e


@app.get("/api/events")
def events() -> dict[str, Any]:
    return {"events": state.events_list()}


@app.get("/api/events.csv")
def events_csv() -> StreamingResponse:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["id", "event", "type", "label", "detail", "video_time_s", "wall_time"])
    for ev in reversed(state.events_list()):
        w.writerow([ev["id"], ev["event"], ev["type"], ev["label"], ev["detail"], ev["video_time"],
                    datetime.fromtimestamp(ev["wall_time"]).isoformat(timespec="seconds")])
    buf.seek(0)
    return StreamingResponse(iter([buf.getvalue()]), media_type="text/csv",
                             headers={"Content-Disposition": "attachment; filename=events.csv"})


# ---------------------------------------------------------------- streams
@app.get("/stream.mjpg")
async def stream() -> StreamingResponse:
    boundary = b"--frame"
    placeholder = (STATIC / "placeholder.jpg").read_bytes() if (STATIC / "placeholder.jpg").exists() else None

    async def gen():
        last_seq = -1
        while not _shutting_down.is_set():
            jpeg, seq = await asyncio.to_thread(state.wait_for_frame, last_seq, 0.5)
            if seq == last_seq or jpeg is None:
                if jpeg is None and placeholder is not None and last_seq == -1:
                    yield boundary + b"\r\nContent-Type: image/jpeg\r\n\r\n" + placeholder + b"\r\n"
                continue
            last_seq = seq
            yield boundary + b"\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(jpeg)).encode() \
                + b"\r\n\r\n" + jpeg + b"\r\n"

    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame",
                             headers={"Cache-Control": "no-cache"})


@app.websocket("/ws")
async def ws(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        events = state.events_list()
        last_id = events[0]["id"] if events else 0  # per-connection cursor: every client sees every event
        await websocket.send_json({"kind": "init", **state.snapshot(), "history": state.history_list(),
                                   "events": events})
        while not _shutting_down.is_set():
            await asyncio.sleep(0.2)
            snap = state.snapshot()
            for ev in state.events_since(last_id):
                last_id = ev["id"]
                await websocket.send_json({"kind": "event", "event": ev})
            await websocket.send_json({"kind": "metrics", **snap})
    except (WebSocketDisconnect, RuntimeError):
        return
