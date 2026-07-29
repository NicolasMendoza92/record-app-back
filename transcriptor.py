"""
Transcripción en la nube + generación de acta con Claude.

Capa de STT agnóstica del proveedor:
  transcribir_audio(audio_bytes, n_speakers) -> ResultadoSTT

El proveedor activo se elige con la variable de entorno STT_PROVIDER
(por defecto "assemblyai"). AssemblyAI es el proveedor primario; Deepgram
queda stubeado para agregarse más adelante.
"""

import os
import sys
from dataclasses import dataclass
from datetime import datetime

try:
    import anthropic
    from dotenv import load_dotenv
    from rich.console import Console
except ImportError as e:
    print(f"\n[ERROR] Falta instalar dependencias: {e}")
    print("Corré: pip install -r requirements.txt\n")
    sys.exit(1)

load_dotenv()
console = Console()

# ── configuración ─────────────────────────────────────────────────────────────
ANTHROPIC_KEY   = os.getenv("ANTHROPIC_API_KEY")
STT_PROVIDER    = os.getenv("STT_PROVIDER", "assemblyai").lower()
LANGUAGE_CODE   = "es"

# ── helpers de consola ────────────────────────────────────────────────────────
def step(msg: str):  console.print(f"\n[cyan]▶[/cyan] {msg}")
def ok(msg: str):    console.print(f"[green]✓[/green] {msg}")


# ── resultado de la capa STT ──────────────────────────────────────────────────
@dataclass
class ResultadoSTT:
    texto: str            # transcripción ya formateada "SPEAKER A: <texto>"
    n_speakers: int       # cantidad de hablantes detectada por el proveedor
    duracion_seg: float   # duración del audio en segundos


# ── capa de transcripción (agnóstica del proveedor) ───────────────────────────
def transcribir_audio(audio_path: str, n_speakers: int) -> ResultadoSTT:
    """
    Transcribe el audio con el proveedor activo (STT_PROVIDER) y devuelve el
    texto formateado por hablante junto con metadata (speakers, duración).

    Recibe la RUTA a un archivo temporal (no los bytes) para no cargar el audio
    entero en memoria; el SDK del proveedor lo sube desde el disco.
    """
    if STT_PROVIDER == "assemblyai":
        return _transcribir_assemblyai(audio_path, n_speakers)
    if STT_PROVIDER == "deepgram":
        return _transcribir_deepgram(audio_path, n_speakers)
    raise ValueError(
        f"STT_PROVIDER desconocido: '{STT_PROVIDER}'. Usá 'assemblyai' o 'deepgram'."
    )


# ── proveedor primario: AssemblyAI ────────────────────────────────────────────
def _transcribir_assemblyai(audio_path: str, n_speakers: int) -> ResultadoSTT:
    """
    AssemblyAI: consciente del solapamiento y acepta el número de hablantes
    esperado (speakers_expected) para mejorar la diarización. El SDK sube el
    audio, lanza la transcripción y pollea hasta que termina.
    """
    import assemblyai as aai

    api_key = os.getenv("ASSEMBLYAI_API_KEY")
    if not api_key:
        raise RuntimeError("ASSEMBLYAI_API_KEY no está configurada en el entorno")
    aai.settings.api_key = api_key

    step(f"Transcribiendo con AssemblyAI (es · {n_speakers} hablantes esperados)...")

    config = aai.TranscriptionConfig(
        language_code=LANGUAGE_CODE,
        speaker_labels=True,
        speakers_expected=n_speakers,   # la reunión sabe cuántos son; se usa tal cual
    )

    transcript = aai.Transcriber().transcribe(audio_path, config)

    if transcript.status == aai.TranscriptStatus.error:
        raise RuntimeError(f"AssemblyAI falló la transcripción: {transcript.error}")

    # Construimos el texto recorriendo las utterances -> "SPEAKER A: <texto>".
    # Si la diarización sale imperfecta no importa: el texto y los nombres
    # propios dichos en voz alta son lo que vale para el acta.
    speakers: set[str] = set()
    lineas: list[str] = []
    for u in (transcript.utterances or []):
        speakers.add(u.speaker)
        lineas.append(f"SPEAKER {u.speaker}: {u.text.strip()}")

    if lineas:
        texto = "\n".join(lineas)
        n_detectados = len(speakers)
    else:
        # sin diarización disponible: caemos al texto plano
        texto = (transcript.text or "").strip()
        n_detectados = n_speakers

    duracion_seg = float(transcript.audio_duration or 0)

    ok(f"Transcripción lista: {len(lineas)} turnos · {n_detectados} hablantes detectados")
    return ResultadoSTT(texto=texto, n_speakers=n_detectados, duracion_seg=duracion_seg)


# ── proveedor secundario: Deepgram (stub) ─────────────────────────────────────
def _transcribir_deepgram(audio_path: str, n_speakers: int) -> ResultadoSTT:
    """
    Placeholder para agregar Deepgram como segundo proveedor.
    Nota: Deepgram NO acepta un número de hablantes esperado, así que
    'n_speakers' se ignoraría en su implementación.
    """
    raise NotImplementedError(
        "Proveedor 'deepgram' todavía no implementado. "
        "Usá STT_PROVIDER=assemblyai por ahora."
    )


# ── generación del acta con Claude (Sonnet) ───────────────────────────────────
SYSTEM_PROMPT = """Sos un asistente que redacta actas de reunión claras y fieles para organizaciones del tercer sector (ONGs, fundaciones).

Trabajás a partir de una transcripción automática de una reunión tipo mesa redonda en español, donde las líneas vienen etiquetadas como "SPEAKER A", "SPEAKER B", etc. Esas etiquetas son solo una ayuda de diarización y pueden estar equivocadas: NO son los responsables de las tareas.

Reglas absolutas:
- Ceñite ÚNICAMENTE a lo que dice la transcripción. No inventes datos, nombres, decisiones ni palabras que no estén en el texto.
- Si un fragmento es inaudible, ambiguo o no se entiende, marcalo como "[inaudible]" en lugar de rellenar o suponer.
- Los responsables de las tareas se deducen por el NOMBRE PROPIO mencionado en voz alta dentro del texto (ej.: "Paula queda en averiguar el caso B" → responsable: Paula), NUNCA por la etiqueta de speaker.
- Si una acción no tiene un responsable claro en el texto, poné "sin asignar".
- Escribí en español correcto, formal pero accesible.
- Usá Markdown.
"""

ACTA_PROMPT = """A continuación, la transcripción automática de la reunión (etiquetas SPEAKER = diarización aproximada, NO responsables):

{transcripcion}

---

Generá un acta en Markdown con exactamente esta estructura:

# Acta de reunión
**Fecha:** {fecha}
**Duración estimada:** {duracion}
**Participantes detectados:** {n_speakers}

## Resumen por tema
(Organizá lo hablado por tema, un subtítulo o párrafo por tema. Solo lo que aparece en la transcripción.)

## Acciones siguientes
(Lista con viñetas. Formato "Nombre: tarea", donde Nombre es el nombre propio mencionado en el texto como responsable. Si no hay responsable claro, usá "sin asignar: tarea". Solo acciones concretas que surjan del texto.)

## Notas adicionales
(Solo si hay algo relevante que no entra arriba; si no, omití esta sección. Marcá lo dudoso como [inaudible].)
"""

# Ancla del ACTA_PROMPT: el bloque de asistentes se inserta JUSTO antes de esta
# línea (dentro del prompt de usuario), sólo cuando hay nombres.
_ANCLA_INSTRUCCIONES = "Generá un acta en Markdown con exactamente esta estructura:"


def _bloque_nombres(nombres: list[str]) -> str:
    """Sección delimitada con la lista de asistentes y las reglas para usarla."""
    lista = "\n".join(f"- {n}" for n in nombres)
    return (
        "=== ASISTENTES A LA REUNIÓN (lista provista por el usuario) ===\n"
        "Nombres propios de quienes asistieron:\n"
        f"{lista}\n\n"
        "Reglas para usar esta lista al redactar el acta:\n"
        "1. ORTOGRAFÍA CANÓNICA: usá exactamente estas grafías al nombrar responsables. "
        "Si en la transcripción aparece un diminutivo o una variante fonética que claramente "
        "corresponde a alguien de la lista, normalizalo a la grafía de la lista.\n"
        "2. LA LISTA NO ES OBLIGATORIA: es la lista de quienes asistieron, no de quienes tienen "
        "tareas. Si una acción no tiene un responsable claro en la transcripción, poné "
        "\"sin asignar\". NO asignes una acción a alguien de la lista sólo porque su nombre está "
        "disponible. Es preferible \"sin asignar\" que un responsable equivocado.\n"
        "3. LA LISTA NO ES CERRADA: si en el audio se menciona a alguien que no está en la lista, "
        "usalo igual tal como se escuchó y marcalo con [no listado].\n\n"
        "Sobre las etiquetas SPEAKER: asociá una etiqueta SPEAKER a un nombre real SÓLO cuando "
        "haya evidencia explícita en el audio (alguien se presenta, o lo interpelan por su nombre). "
        "Si no hay evidencia, dejá la etiqueta SPEAKER como está. Está prohibido inferir el nombre "
        "por el orden de aparición o por la cantidad de participación.\n"
        "=== FIN ASISTENTES ===\n"
    )


def generar_acta(
    transcripcion: str,
    duracion_min: int,
    n_speakers: int,
    nombres: list[str] | None = None,
) -> str:
    """Genera el acta estructurada en Markdown con Claude (Sonnet).

    'nombres' es opcional: la lista (saneada) de asistentes provista por el
    usuario. Si viene vacía o None, el prompt queda EXACTAMENTE como sin la
    feature (no se agrega ninguna sección).
    """
    step("Generando acta estructurada con Claude...")

    if not ANTHROPIC_KEY:
        raise RuntimeError("ANTHROPIC_API_KEY no está configurada en el entorno")

    client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)

    prompt_usuario = ACTA_PROMPT.format(
        transcripcion=transcripcion,
        fecha=datetime.now().strftime("%d/%m/%Y"),
        duracion=f"~{duracion_min} minutos",
        n_speakers=n_speakers,
    )

    # Sólo si hay nombres: insertamos el bloque antes de las instrucciones de
    # estructura. Sin nombres, prompt_usuario queda idéntico al de siempre.
    if nombres:
        prompt_usuario = prompt_usuario.replace(
            _ANCLA_INSTRUCCIONES,
            _bloque_nombres(nombres) + "\n" + _ANCLA_INSTRUCCIONES,
            1,
        )

    message = client.messages.create(
        model="claude-sonnet-5",          # Sonnet vigente (el id 4-20250514 devuelve 404)
        max_tokens=2048,
        thinking={"type": "disabled"},    # salida directa; los 2048 tokens quedan para el acta
        system=SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": prompt_usuario,
        }],
    )

    acta = message.content[0].text
    ok("Acta generada")
    return acta
