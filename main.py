"""
Backend FastAPI — expone el pipeline de transcripción como API REST
Archivo: back/main.py  (el Procfile y Render apuntan a este)
"""

import os
import re
import json
import time
import uuid
import shutil
import secrets
import tempfile
import logging
import threading
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware

from transcriptor import transcribir_audio, generar_acta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("transcriptor")

# saltos de línea y caracteres de control (los nombres van a un prompt)
_CARACTERES_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
MAX_NOMBRES = 20
MAX_LARGO_NOMBRE = 60


def sanear_nombres(raw: str) -> list[str]:
    """Parsea y sanea la lista de asistentes que llega como JSON en el FormData.

    Nunca lanza: si el JSON viene malformado o no es una lista de strings,
    loguea un warning y devuelve []. El audio ya se subió (es lo caro), así que
    jamás fallamos la request entera por este campo opcional.
    """
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        logger.warning("Campo 'nombres' malformado; se ignora y se sigue con lista vacía")
        return []

    if not isinstance(data, list):
        logger.warning("Campo 'nombres' no es una lista; se ignora")
        return []

    limpios: list[str] = []
    vistos: set[str] = set()
    for item in data:
        if not isinstance(item, str):
            continue
        # sacar control chars/saltos de línea y colapsar espacios repetidos
        s = " ".join(_CARACTERES_CONTROL.sub(" ", item).split()).strip()
        if not s:
            continue
        s = s[:MAX_LARGO_NOMBRE]
        clave = s.lower()
        if clave in vistos:
            continue
        vistos.add(clave)
        limpios.append(s)
        if len(limpios) >= MAX_NOMBRES:
            break
    return limpios

app = FastAPI(title="Transcriptor API", version="2.0.0")

# Orígenes permitidos. Local: http://localhost:3000. En Render, setear la env var
# ALLOWED_ORIGINS al dominio de PRODUCCIÓN de Vercel (varios, separados por coma).
ALLOWED_ORIGINS = [
    o.strip()
    for o in os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
    if o.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    # deploys de preview de Vercel: record-app-front-<hash>.vercel.app
    allow_origin_regex=r"https://record-app-front-[a-z0-9-]+\.vercel\.app",
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["X-App-Password", "Content-Type"],
    allow_credentials=False,
    max_age=3600,
)


def verificar_password(x_app_password: str = Header(default="")):
    """Auth como DEPENDENCIA (no middleware).

    En Starlette los middlewares agregados después envuelven a los anteriores.
    Como dependencia, CORSMiddleware envuelve esto y hasta un 401 sale CON headers
    CORS (si no, el browser lo reporta como error de CORS en vez de 401).
    Comparación constant-time con secrets.compare_digest.
    """
    esperada = os.environ.get("APP_PASSWORD", "")
    if not esperada or not secrets.compare_digest(x_app_password, esperada):
        raise HTTPException(status_code=401, detail="No autorizado")


# ── almacén de jobs en memoria ──────────────────────────────────────────────
# La free tier de Render es UNA sola instancia sin autoscaling, así que un dict
# en memoria alcanza. Un redeploy pierde los jobs en vuelo — aceptable y esperado.
# Cada job: {estado, resultado, error, actualizado}.
# estado ∈ subiendo | transcribiendo | redactando | listo | error
JOBS: dict[str, dict] = {}
_JOBS_LOCK = threading.Lock()
_JOB_TTL_SEG = 30 * 60  # se limpian los jobs terminados a los 30 min


def _limpiar_jobs() -> None:
    """Borra jobs terminados (listo/error) más viejos que el TTL. Evita fugar memoria."""
    corte = time.time() - _JOB_TTL_SEG
    with _JOBS_LOCK:
        muertos = [
            jid for jid, j in JOBS.items()
            if j["estado"] in ("listo", "error") and j["actualizado"] < corte
        ]
        for jid in muertos:
            JOBS.pop(jid, None)


def _set_job(jid: str, **campos) -> None:
    with _JOBS_LOCK:
        j = JOBS.get(jid)
        if j is not None:
            j.update(campos)
            j["actualizado"] = time.time()


def _procesar_job(jid: str, audio_path: str, speakers: int, nombres: list[str]) -> None:
    """Corre en un thread aparte: transcribe + redacta el acta y va actualizando el job.

    El SDK de AssemblyAI y la llamada a Claude son bloqueantes; por eso esto vive
    en un thread y no en el event loop. El endpoint ya respondió 202.
    """
    try:
        _set_job(jid, estado="transcribiendo")
        stt = transcribir_audio(audio_path, n_speakers=speakers)

        duracion_min = max(1, int(stt.duracion_seg // 60)) if stt.duracion_seg else 0

        _set_job(jid, estado="redactando")
        acta = generar_acta(
            stt.texto,
            duracion_min=duracion_min,
            n_speakers=stt.n_speakers,
            nombres=nombres,   # opcional; [] deja el prompt como siempre
        )

        _set_job(jid, estado="listo", resultado={
            "transcripcion": stt.texto,
            "acta": acta,
            "duracion_min": duracion_min,
            "n_speakers": stt.n_speakers,   # cantidad detectada por el proveedor
        })

    except Exception as e:
        logger.exception("Fallo procesando job %s", jid)
        _set_job(jid, estado="error", error=f"{type(e).__name__}: {e}")

    finally:
        # el archivo temporal se borra pase lo que pase
        try:
            os.remove(audio_path)
        except OSError:
            pass


@app.get("/health")
def health():
    # sin auth: keepalive del front + healthcheck de Render
    return {"status": "ok"}


@app.get("/verificar", dependencies=[Depends(verificar_password)])
def verificar():
    """Chequeo de contraseña para el login del front: 200 si es correcta, 401 si no."""
    return {"ok": True}


@app.post("/transcribir", status_code=202, dependencies=[Depends(verificar_password)])
async def endpoint_transcribir(
    audio: UploadFile = File(...),
    speakers: int = Form(default=2, ge=1, le=10),
    nombres: str = Form(default="[]"),
):
    """Arranca el job y devuelve 202 con el job_id. El trabajo pesado corre en un
    thread; el front poletea GET /transcribir/{job_id} hasta que esté 'listo'."""
    extensiones_ok = {".mp3", ".m4a", ".mp4", ".wav", ".ogg", ".flac", ".webm", ".opus"}
    ext = Path(audio.filename or "").suffix.lower()
    if ext not in extensiones_ok:
        raise HTTPException(status_code=400, detail=f"Formato no soportado: {ext}")

    # El audio se vuelca a disco DENTRO de la request (el UploadFile se cierra al
    # responder); el thread trabaja después sobre esa ruta. Sin cargarlo a memoria.
    audio.file.seek(0)
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext or ".tmp") as tmp:
        shutil.copyfileobj(audio.file, tmp)
        audio_path = tmp.name

    nombres_asistentes = sanear_nombres(nombres)  # [] si no vino nada o vino malformado

    _limpiar_jobs()
    jid = uuid.uuid4().hex
    with _JOBS_LOCK:
        JOBS[jid] = {"estado": "subiendo", "resultado": None, "error": None, "actualizado": time.time()}

    threading.Thread(
        target=_procesar_job,
        args=(jid, audio_path, speakers, nombres_asistentes),
        daemon=True,
    ).start()

    return {"job_id": jid}


@app.get("/transcribir/{job_id}", dependencies=[Depends(verificar_password)])
def estado_job(job_id: str):
    """Estado del job para el polling del front."""
    with _JOBS_LOCK:
        j = JOBS.get(job_id)
        if j is None:
            raise HTTPException(status_code=404, detail="Job desconocido o expirado")
        return {"estado": j["estado"], "resultado": j["resultado"], "error": j["error"]}
