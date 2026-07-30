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
# Cada job: {estado, resultado, error, actualizado, + transcripcion/duracion_min/
# n_speakers/nombres una vez transcripto (para no perderlo si el acta falla)}.
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


def _borrar_archivo(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


def _procesar_job(jid: str, audio_path: str, speakers: int, nombres: list[str]) -> None:
    """Corre en un thread aparte: transcribe y luego redacta el acta.

    Apenas la transcripción está lista se GUARDA en el job. Así, si el acta falla,
    no se pierde la transcripción (que ya se pagó) y se puede reintentar sólo el
    acta sin volver a llamar a AssemblyAI. El SDK y Claude son bloqueantes; por eso
    esto vive en un thread y no en el event loop. El endpoint ya respondió 202.
    """
    try:
        _set_job(jid, estado="transcribiendo")
        stt = transcribir_audio(audio_path, n_speakers=speakers)
        duracion_min = max(1, int(stt.duracion_seg // 60)) if stt.duracion_seg else 0
        # ── checkpoint: la transcripción queda persistida en el job ──
        _set_job(
            jid,
            transcripcion=stt.texto,
            duracion_min=duracion_min,
            n_speakers=stt.n_speakers,   # cantidad detectada por el proveedor
            nombres=nombres,             # se guarda para poder reintentar el acta
        )
    except Exception as e:
        logger.exception("Fallo transcribiendo job %s", jid)
        _set_job(jid, estado="error", error=f"{type(e).__name__}: {e}")
        _borrar_archivo(audio_path)
        return

    # el audio ya no se necesita: el reintento sólo re-redacta sobre la transcripción
    _borrar_archivo(audio_path)
    _redactar_acta(jid)


def _redactar_acta(jid: str) -> None:
    """Redacta el acta a partir de la transcripción ya guardada en el job.

    Se usa tanto en el flujo normal como en el reintento. No re-transcribe nada.
    """
    with _JOBS_LOCK:
        j = JOBS.get(jid)
        if j is None or j.get("transcripcion") is None:
            return
        transcripcion = j["transcripcion"]
        duracion_min = j["duracion_min"]
        n_speakers = j["n_speakers"]
        nombres = j.get("nombres") or []

    _set_job(jid, estado="redactando", error=None)
    try:
        acta = generar_acta(
            transcripcion,
            duracion_min=duracion_min,
            n_speakers=n_speakers,
            nombres=nombres,   # opcional; [] deja el prompt como siempre
        )
        _set_job(jid, estado="listo", error=None, resultado={
            "transcripcion": transcripcion,
            "acta": acta,
            "duracion_min": duracion_min,
            "n_speakers": n_speakers,
        })
    except Exception as e:
        # la transcripción sigue guardada en el job: no se pierde, se puede reintentar
        logger.exception("Fallo redactando acta del job %s", jid)
        _set_job(jid, estado="error", error=f"{type(e).__name__}: {e}")


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
        JOBS[jid] = {
            "estado": "subiendo",
            "resultado": None,
            "error": None,
            "actualizado": time.time(),
            # se completan al terminar la transcripción (checkpoint anti-pérdida)
            "transcripcion": None,
            "duracion_min": None,
            "n_speakers": None,
            "nombres": nombres_asistentes,
        }

    threading.Thread(
        target=_procesar_job,
        args=(jid, audio_path, speakers, nombres_asistentes),
        daemon=True,
    ).start()

    return {"job_id": jid}


@app.get("/transcribir/{job_id}", dependencies=[Depends(verificar_password)])
def estado_job(job_id: str):
    """Estado del job para el polling del front.

    Incluye 'parcial' con la transcripción apenas está disponible: si el acta
    falla, el front la muestra igual (no se pierde lo ya transcripto/pagado).
    """
    with _JOBS_LOCK:
        j = JOBS.get(job_id)
        if j is None:
            raise HTTPException(status_code=404, detail="Job desconocido o expirado")
        parcial = None
        if j.get("transcripcion") is not None:
            parcial = {
                "transcripcion": j["transcripcion"],
                "duracion_min": j["duracion_min"],
                "n_speakers": j["n_speakers"],
            }
        return {
            "estado": j["estado"],
            "resultado": j["resultado"],
            "error": j["error"],
            "parcial": parcial,
        }


@app.post("/transcribir/{job_id}/acta", status_code=202, dependencies=[Depends(verificar_password)])
def reintentar_acta(job_id: str):
    """Reintenta SOLO la redacción del acta sobre la transcripción ya guardada.

    No re-transcribe (no re-cobra AssemblyAI). El front vuelve a poletear el job.
    """
    with _JOBS_LOCK:
        j = JOBS.get(job_id)
        if j is None:
            raise HTTPException(status_code=404, detail="Job desconocido o expirado")
        if j.get("transcripcion") is None:
            raise HTTPException(status_code=409, detail="El job no tiene una transcripción para redactar")
        if j["estado"] in ("transcribiendo", "redactando"):
            raise HTTPException(status_code=409, detail="El job ya está en curso")

    threading.Thread(target=_redactar_acta, args=(job_id,), daemon=True).start()
    return {"job_id": job_id}
