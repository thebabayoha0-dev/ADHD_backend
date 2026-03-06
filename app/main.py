import asyncio
import io
import json
import logging
import os
import uuid
import zipfile
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
DATA_DIR = os.getenv("DATA_DIR", "./data")
os.makedirs(DATA_DIR, exist_ok=True)

def _parse_origins() -> List[str]:
    raw = os.getenv("CORS_ORIGINS", "http://localhost:5173")
    return [x.strip() for x in raw.split(",") if x.strip()]

# Logging (stdout is best for containers)
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("adhd-eye-backend")

# -----------------------------------------------------------------------------
# App
# -----------------------------------------------------------------------------
app = FastAPI(title="ADHD Eye Tracking Backend", version="1.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_parse_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"ok": True, "ts": datetime.now(timezone.utc).isoformat()}

# -----------------------------------------------------------------------------
# Session Store
# -----------------------------------------------------------------------------
# NOTE: This is still in-memory for fast access, but we also stream every WS
# message into JSONL files per session so a single crash doesn't lose data.
# For true production (multi-replica): move to Redis/Postgres + object storage.

class SessionStart(BaseModel):
    participantId: str = Field(min_length=1, max_length=64)
    age: int = Field(ge=3, le=18)
    condition: str = Field(default="Unsure", max_length=32)

class SessionState(BaseModel):
    meta: Dict[str, Any]
    events: List[Dict[str, Any]] = []
    gaze: List[Dict[str, Any]] = []
    calib: List[Dict[str, Any]] = []
    video_path: Optional[str] = None

SESSIONS: Dict[str, SessionState] = {}
SESSION_LOCK = asyncio.Lock()

def _session_dir(session_id: str) -> str:
    out_dir = os.path.join(DATA_DIR, session_id)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir

def _append_jsonl(session_id: str, filename: str, obj: Dict[str, Any]) -> None:
    path = os.path.join(_session_dir(session_id), filename)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

@app.post("/session/start")
async def start_session(meta: SessionStart):
    sid = str(uuid.uuid4())
    async with SESSION_LOCK:
        SESSIONS[sid] = SessionState(meta=meta.model_dump())
    _append_jsonl(sid, "meta.jsonl", {"t": datetime.now(timezone.utc).isoformat(), **meta.model_dump()})
    return {"sessionId": sid}

@app.post("/session/{session_id}/upload_video")
async def upload_video(session_id: str, file: UploadFile = File(...)):
    async with SESSION_LOCK:
        s = SESSIONS.get(session_id)
    if not s:
        raise HTTPException(404, "Unknown session")

    payload = await file.read()
    if not payload:
        raise HTTPException(400, "Uploaded video is empty")

    out_path = os.path.join(_session_dir(session_id), "webcam.webm")
    with open(out_path, "wb") as f:
        f.write(payload)

    s.video_path = out_path
    log.info("Saved webcam video for session %s (%d bytes)", session_id, len(payload))
    return {"ok": True, "bytes": len(payload)}

@app.get("/session/{session_id}/download_zip")
async def download_zip(session_id: str):
    async with SESSION_LOCK:
        s = SESSIONS.get(session_id)
    if not s:
        raise HTTPException(404, "Unknown session")

    out_dir = _session_dir(session_id)

    # Save raw logs
    events_df = pd.DataFrame(s.events)
    gaze_df = pd.DataFrame(s.gaze)
    calib_df = pd.DataFrame(s.calib)

    events_csv = os.path.join(out_dir, "events.csv")
    gaze_csv = os.path.join(out_dir, "gaze.csv")
    calib_csv = os.path.join(out_dir, "calibration.csv")

    events_df.to_csv(events_csv, index=False)
    gaze_df.to_csv(gaze_csv, index=False)
    calib_df.to_csv(calib_csv, index=False)

    # Feature extraction
    features = extract_features(events_df, gaze_df)
    pros = features.get("prosaccade_features", pd.DataFrame())
    anti = features.get("antisaccade_features", pd.DataFrame())
    fix = features.get("fixation_features", pd.DataFrame())
    purs = features.get("smooth_pursuit_features", pd.DataFrame())
    rt = features.get("reaction_time", pd.DataFrame())

    pros_path = os.path.join(out_dir, "prosaccade_features.csv")
    anti_path = os.path.join(out_dir, "antisaccade_features.csv")
    fix_path = os.path.join(out_dir, "fixation_features.csv")
    purs_path = os.path.join(out_dir, "smooth_pursuit_features.csv")
    rt_path = os.path.join(out_dir, "reaction_time.csv")

    pros.to_csv(pros_path, index=False)
    anti.to_csv(anti_path, index=False)
    fix.to_csv(fix_path, index=False)
    purs.to_csv(purs_path, index=False)
    rt.to_csv(rt_path, index=False)

    # Zip in-memory
    zbuf = io.BytesIO()
    with zipfile.ZipFile(zbuf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        if s.video_path and os.path.exists(s.video_path):
            z.write(s.video_path, arcname="recorded_webcam.webm")
        z.write(events_csv, arcname="events.csv")
        z.write(gaze_csv, arcname="gaze.csv")
        z.write(calib_csv, arcname="calibration.csv")
        z.write(pros_path, arcname="prosaccade_features.csv")
        z.write(anti_path, arcname="antisaccade_features.csv")
        z.write(fix_path, arcname="fixation_features.csv")
        z.write(purs_path, arcname="smooth_pursuit_features.csv")
        z.write(rt_path, arcname="reaction_time.csv")
        z.writestr("session_meta.json", json.dumps(s.meta, indent=2))

    zbuf.seek(0)
    return StreamingResponse(
        zbuf,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{session_id}_adhd_eye_tasks.zip"'},
    )

# -----------------------------------------------------------------------------
# WebSocket Ingest
# -----------------------------------------------------------------------------
@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    session_id: Optional[str] = None

    try:
        while True:
            msg = await ws.receive_text()
            try:
                data = json.loads(msg)
            except Exception:
                await ws.send_text(json.dumps({"type": "error", "message": "Invalid JSON"}))
                continue

            mtype = data.get("type")
            session_id = data.get("sessionId") or session_id

            # Allow explicit bind
            if mtype == "bind":
                sid = data.get("sessionId")
                if sid in SESSIONS:
                    session_id = sid
                    await ws.send_text(json.dumps({"type": "ack", "sessionId": sid}))
                else:
                    await ws.send_text(json.dumps({"type": "error", "message": "Unknown sessionId"}))
                continue

            # Must have a valid session id for data messages
            if not session_id or session_id not in SESSIONS:
                # Ignore until bound / valid sid included
                continue

            async with SESSION_LOCK:
                s = SESSIONS[session_id]

            t = data.get("t_ms")

            if mtype == "hello":
                _append_jsonl(session_id, "ws.jsonl", {"type": "hello", "t_ms": t, "meta": data.get("meta")})
                await ws.send_text(json.dumps({"type": "ack", "sessionId": session_id}))
                continue

            if mtype == "event":
                row = {"t_ms": t, "name": data.get("name"), "payload": data.get("payload")}
                s.events.append(row)
                _append_jsonl(session_id, "events.jsonl", row)

            elif mtype == "gaze":
                row = {
                    "t_ms": t,
                    "x": data.get("x"),
                    "y": data.get("y"),
                    "valid": data.get("valid"),
                    "raw": json.dumps(data.get("raw", {})),
                }
                s.gaze.append(row)
                _append_jsonl(session_id, "gaze.jsonl", row)

            elif mtype == "calib":
                row = {
                    "t_ms": t,
                    "screen_x": data.get("screen_x"),
                    "screen_y": data.get("screen_y"),
                    "gaze_x": data.get("gaze_x"),
                    "gaze_y": data.get("gaze_y"),
                    "valid": data.get("valid"),
                }
                s.calib.append(row)
                _append_jsonl(session_id, "calibration.jsonl", row)

            elif mtype == "done":
                _append_jsonl(session_id, "ws.jsonl", {"type": "done", "t_ms": t})
                await ws.send_text(json.dumps({"type": "ack", "sessionId": session_id}))
                break

    except WebSocketDisconnect:
        return

# -----------------------------------------------------------------------------
# Feature extraction (baseline, research-friendly)
# -----------------------------------------------------------------------------

def extract_features(events: pd.DataFrame, gaze: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}

    # ---- utilities to tolerate minor schema drift in event logs ----
    def _pick_time_col(df: pd.DataFrame) -> Optional[str]:
        """Return the best guess time column name, or None."""
        for c in ("t_ms", "t", "timestamp_ms", "ts_ms", "time_ms"):
            if c in df.columns:
                return c
        return None

    def _ensure_col(df: pd.DataFrame, col: str, default=np.nan) -> None:
        if col not in df.columns:
            df[col] = default

    if events.empty:
        out["prosaccade_features"] = pd.DataFrame()
        out["antisaccade_features"] = pd.DataFrame()
        out["fixation_features"] = pd.DataFrame()
        out["smooth_pursuit_features"] = pd.DataFrame()
        out["reaction_time"] = pd.DataFrame()
        return out

    def payload_get(row, key, default=None):
        try:
            p = row.get("payload")
            if isinstance(p, str):
                p = json.loads(p)
            if isinstance(p, dict):
                return p.get(key, default)
        except Exception:
            return default
        return default

    # Reaction time
    resp = events[events["name"] == "response"].copy()
    if not resp.empty:
        resp["game"] = resp.apply(lambda r: payload_get(r, "game", ""), axis=1)
        resp["trial_id"] = resp.apply(lambda r: payload_get(r, "trial_id", ""), axis=1)
        resp["correct"] = resp.apply(lambda r: payload_get(r, "correct", ""), axis=1)
        resp["rt_ms"] = resp.apply(lambda r: payload_get(r, "rt_ms", ""), axis=1)
        out["reaction_time"] = resp[["t_ms", "game", "trial_id", "correct", "rt_ms"]].reset_index(drop=True)
    else:
        out["reaction_time"] = pd.DataFrame(columns=["t_ms", "game", "trial_id", "correct", "rt_ms"])

    # Prosaccade (game 2)
    tgt = events[events["name"] == "target_on"].copy()
    if not tgt.empty:
        tgt["game"] = tgt.apply(lambda r: payload_get(r, "game", ""), axis=1)
        pros_tgt = tgt[tgt["game"] == 2].copy()

        if not resp.empty and not pros_tgt.empty:
            pros_tgt["trial_id"] = pros_tgt.apply(lambda r: payload_get(r, "trial_id", ""), axis=1)
            resp2 = resp[resp.apply(lambda r: payload_get(r, "game", "") == 2, axis=1)].copy()
            resp2["trial_id"] = resp2.apply(lambda r: payload_get(r, "trial_id", ""), axis=1)
            resp2["rt_ms"] = resp2.apply(lambda r: payload_get(r, "rt_ms", np.nan), axis=1)
            resp2["correct"] = resp2.apply(lambda r: payload_get(r, "correct", np.nan), axis=1)

            # tolerate different time column names in target events
            tcol = _pick_time_col(pros_tgt)
            if tcol and tcol != "target_on_t_ms":
                pros_tgt = pros_tgt.rename(columns={tcol: "target_on_t_ms"})
            _ensure_col(pros_tgt, "target_on_t_ms", np.nan)

            m = pros_tgt.merge(resp2[["trial_id", "rt_ms", "correct"]], on="trial_id", how="left")
            m["side"] = m.apply(lambda r: payload_get(r, "side", ""), axis=1)
            m["isTarget"] = m.apply(lambda r: payload_get(r, "isTarget", ""), axis=1)
            out["prosaccade_features"] = (
                m[["trial_id", "target_on_t_ms", "side", "isTarget", "rt_ms", "correct"]]
                .reset_index(drop=True)
            )
        else:
            out["prosaccade_features"] = pd.DataFrame(
                columns=["trial_id", "target_on_t_ms", "side", "isTarget", "rt_ms", "correct"]
            )
    else:
        out["prosaccade_features"] = pd.DataFrame(columns=["trial_id", "target_on_t_ms", "side", "isTarget", "rt_ms", "correct"])

    # Antisaccade (game 3)
    cue = events[events["name"] == "cue_on"].copy()
    tgt3 = tgt[tgt.apply(lambda r: payload_get(r, "game", "") == 3, axis=1)].copy() if not tgt.empty else pd.DataFrame()
    resp3 = resp[resp.apply(lambda r: payload_get(r, "game", "") == 3, axis=1)].copy() if not resp.empty else pd.DataFrame()

    if not cue.empty and not tgt3.empty:
        cue3 = cue[cue.apply(lambda r: payload_get(r, "game", "") == 3, axis=1)].copy()
        cue3["trial_id"] = cue3.apply(lambda r: payload_get(r, "trial_id", ""), axis=1)
        cue3["cue_side"] = cue3.apply(lambda r: payload_get(r, "side", ""), axis=1)

        tgt3["trial_id"] = tgt3.apply(lambda r: payload_get(r, "trial_id", ""), axis=1)
        tgt3["target_side"] = tgt3.apply(lambda r: payload_get(r, "side", ""), axis=1)

        if not resp3.empty:
            resp3["trial_id"] = resp3.apply(lambda r: payload_get(r, "trial_id", ""), axis=1)
            resp3["rt_ms"] = resp3.apply(lambda r: payload_get(r, "rt_ms", np.nan), axis=1)
            resp3["correct"] = resp3.apply(lambda r: payload_get(r, "correct", np.nan), axis=1)

            # tolerate different time column names in target events
            tcol = _pick_time_col(tgt3)
            if tcol:
                tgt3_small = tgt3[["trial_id", tcol, "target_side"]].rename(columns={tcol: "target_on_t_ms"})
            else:
                tgt3_small = tgt3[["trial_id", "target_side"]].copy()
                tgt3_small["target_on_t_ms"] = np.nan

            m = cue3.merge(tgt3_small[["trial_id", "target_side", "target_on_t_ms"]], on="trial_id", how="left")
            m = m.merge(resp3[["trial_id", "rt_ms", "correct"]], on="trial_id", how="left")
            _ensure_col(m, "target_on_t_ms", np.nan)
            out["antisaccade_features"] = m[["trial_id", "cue_side", "target_side", "target_on_t_ms", "rt_ms", "correct"]].reset_index(drop=True)
        else:
            out["antisaccade_features"] = pd.DataFrame(columns=["trial_id", "cue_side", "target_side", "target_on_t_ms", "rt_ms", "correct"])
    else:
        out["antisaccade_features"] = pd.DataFrame(columns=["trial_id", "cue_side", "target_side", "target_on_t_ms", "rt_ms", "correct"])

    # Helper: game time windows
    def time_window(game_num: int):
        s = events[(events["name"] == "game_start") & events.apply(lambda r: payload_get(r, "game", None) == game_num, axis=1)]
        e = events[(events["name"] == "game_end") & events.apply(lambda r: payload_get(r, "game", None) == game_num, axis=1)]
        if s.empty or e.empty:
            return None
        return float(s.iloc[0]["t_ms"]), float(e.iloc[0]["t_ms"])

    # Fixation (game 4)
    win4 = time_window(4)
    if win4 and not gaze.empty:
        t0, t1 = win4
        g = gaze[(gaze["t_ms"] >= t0) & (gaze["t_ms"] <= t1) & (gaze["valid"] == True)].copy()
        if not g.empty:
            g["x"] = g["x"].astype(float)
            g["y"] = g["y"].astype(float)
            center_x = g["x"].median()
            center_y = g["y"].median()
            disp = np.sqrt((g["x"] - center_x) ** 2 + (g["y"] - center_y) ** 2)
            out["fixation_features"] = pd.DataFrame(
                [
                    {
                        "t_start_ms": int(t0),
                        "t_end_ms": int(t1),
                        "n_samples": int(len(g)),
                        "center_x": float(center_x),
                        "center_y": float(center_y),
                        "std_x": float(g["x"].std()),
                        "std_y": float(g["y"].std()),
                        "dispersion_mean": float(disp.mean()),
                        "dispersion_p95": float(np.percentile(disp, 95)),
                    }
                ]
            )
        else:
            out["fixation_features"] = pd.DataFrame([{ "t_start_ms": int(t0), "t_end_ms": int(t1), "n_samples": 0 }])
    else:
        out["fixation_features"] = pd.DataFrame(columns=["t_start_ms", "t_end_ms", "n_samples", "center_x", "center_y", "std_x", "std_y", "dispersion_mean", "dispersion_p95"])

    # Smooth pursuit (game 5)
    win5 = time_window(5)
    if win5 and not gaze.empty:
        t0, t1 = win5
        tgtpos = events[events["name"] == "pursuit_target_pos"].copy()
        if not tgtpos.empty:
            xs: List[float] = []
            ts: List[float] = []
            for _, r in tgtpos.iterrows():
                p = r.get("payload")
                if isinstance(p, str):
                    try:
                        p = json.loads(p)
                    except Exception:
                        p = {}
                if isinstance(p, dict):
                    if p.get("x") is not None:
                        xs.append(float(p.get("x")))
                        ts.append(float(r["t_ms"]))

            if len(ts) >= 5:
                target_series = pd.DataFrame({"t_ms": ts, "tx": xs}).dropna().sort_values("t_ms")
                g = gaze[(gaze["t_ms"] >= t0) & (gaze["t_ms"] <= t1) & (gaze["valid"] == True)].copy().sort_values("t_ms")
                if not g.empty:
                    tx_interp = np.interp(
                        g["t_ms"].to_numpy(),
                        target_series["t_ms"].to_numpy(),
                        target_series["tx"].to_numpy(),
                    )
                    gx = g["x"].astype(float).to_numpy()
                    corr = float(np.corrcoef(gx, tx_interp)[0, 1]) if (np.std(gx) > 1e-6 and np.std(tx_interp) > 1e-6) else float("nan")
                    rmse = float(np.sqrt(np.mean((gx - tx_interp) ** 2)))
                    out["smooth_pursuit_features"] = pd.DataFrame(
                        [
                            {
                                "t_start_ms": int(t0),
                                "t_end_ms": int(t1),
                                "n_samples": int(len(g)),
                                "corr_gaze_target_x": corr,
                                "rmse_gaze_target_x": rmse,
                            }
                        ]
                    )
                else:
                    out["smooth_pursuit_features"] = pd.DataFrame([{ "t_start_ms": int(t0), "t_end_ms": int(t1), "n_samples": 0 }])
            else:
                out["smooth_pursuit_features"] = pd.DataFrame()
        else:
            out["smooth_pursuit_features"] = pd.DataFrame()
    else:
        out["smooth_pursuit_features"] = pd.DataFrame(columns=["t_start_ms", "t_end_ms", "n_samples", "corr_gaze_target_x", "rmse_gaze_target_x"])

    return out


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("app.main:app", host=host, port=port, reload=False)
