"""Gateway configuration: service URLs and timeouts.

All values can be overridden via environment variables.
"""

import os

STT_URL = os.environ.get("STT_URL", "http://localhost:8001")
TTS_URL = os.environ.get("TTS_URL", "http://localhost:8002")
VBV_URL = os.environ.get("VBV_URL", "http://localhost:8005")
RVC_URL = os.environ.get("RVC_URL", "http://localhost:8003")
LLM_URL = os.environ.get("LLM_URL", "http://localhost:8004")

# httpx timeout in seconds - fish-speech on long text can exceed 5 minutes
try:
    REQUEST_TIMEOUT = float(os.environ.get("GATEWAY_TIMEOUT", "900"))
except ValueError:
    REQUEST_TIMEOUT = 900.0

SERVICES = {
    "stt": STT_URL,
    "tts": TTS_URL,
    "vbv": VBV_URL,
    "rvc": RVC_URL,
    "llm": LLM_URL,
}

ROUTE_MAP = {
    "/transcribe": f"{STT_URL}/transcribe",
    "/tts/fish-speech": f"{TTS_URL}/tts/fish-speech",
    "/tts/vieneu": f"{TTS_URL}/tts/vieneu",
    "/tts/vbv": f"{VBV_URL}/tts/vbv",
    "/convert-voice": f"{RVC_URL}/convert-voice",
    "/llm/chat": f"{LLM_URL}/chat",
}
