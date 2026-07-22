"""
Backend FastAPI — expone el pipeline de transcripción como API REST
Archivo: back/main.py  (el Procfile y Render apuntan a este)
"""

import os
import logging
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from transcriptor import transcribir_audio, generar_acta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("transcriptor")

app = FastAPI(title="Transcriptor API", version="2.0.0")

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


class ResultadoTranscripcion(BaseModel):
    transcripcion: str
    acta: str
    duracion_min: int
    n_speakers: int


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/transcribir", response_model=ResultadoTranscripcion)
async def endpoint_transcribir(
    audio: UploadFile = File(...),
    speakers: int = Form(default=2, ge=1, le=10),
):
    extensiones_ok = {".mp3", ".m4a", ".mp4", ".wav", ".ogg", ".flac"}
    ext = Path(audio.filename or "").suffix.lower()
    if ext not in extensiones_ok:
        raise HTTPException(status_code=400, detail=f"Formato no soportado: {ext}")

    audio_bytes = await audio.read()

    try:
        # 'speakers' del front se usa como número de hablantes esperado.
        stt = transcribir_audio(audio_bytes, n_speakers=speakers)

        duracion_min = max(1, int(stt.duracion_seg // 60)) if stt.duracion_seg else 0

        acta = generar_acta(
            stt.texto,
            duracion_min=duracion_min,
            n_speakers=stt.n_speakers,
        )

        return ResultadoTranscripcion(
            transcripcion=stt.texto,
            acta=acta,
            duracion_min=duracion_min,
            n_speakers=stt.n_speakers,   # cantidad detectada por el proveedor
        )

    except Exception as e:
        # Log del traceback completo en el servidor (antes solo se veía "500").
        logger.exception("Fallo procesando /transcribir")
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")
