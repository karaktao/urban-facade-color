import base64
import logging
from typing import Any, Dict

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from backend.core import (
    analyze_image_bytes,
    encode_png,
    load_model,
)

logger = logging.getLogger(__name__)
app = FastAPI(title="Urban Facade Color API", version="0.1.0")


# Preload model on startup so inference requests pay less cold-start cost.
try:
    load_model()
except Exception as exc:  # noqa: BLE001 - surface error to users via endpoint
    logger.error("Failed to preload model: %s", exc)


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)) -> JSONResponse:
    if file is None:
        raise HTTPException(status_code=400, detail="Image file is required")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    try:
        bgra, colors = analyze_image_bytes(data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected inference error")
        raise HTTPException(status_code=500, detail="Failed to run inference") from exc

    png_bytes = encode_png(bgra)
    b64 = base64.b64encode(png_bytes).decode("utf-8")
    payload: Dict[str, Any] = {
        "image_png_base64": b64,
        "colors": [{"rgb": rgb, "ratio": ratio} for rgb, ratio in colors],
    }
    return JSONResponse(content=payload)