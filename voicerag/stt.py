"""Speech-to-text providers.

Sarvam is the default: MSMARCO-XI is 13 Indic languages, and Saarika is trained
for exactly that, including code-mixed speech. ElevenLabs Scribe is wired up
behind the same interface so the provider is a config switch, not a rewrite.

Both providers return a `Transcript`; the caller never learns which one ran.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)


class STTError(RuntimeError):
    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


@dataclass
class Transcript:
    text: str
    language: str | None = None
    provider: str = "unknown"
    raw: dict | None = None


class BaseSTT:
    name = "base"

    async def transcribe(self, audio: bytes, filename: str, content_type: str) -> Transcript:
        raise NotImplementedError


class SarvamSTT(BaseSTT):
    """POST https://api.sarvam.ai/speech-to-text, auth via `api-subscription-key`.

    The REST endpoint is the synchronous one (clips up to ~30s), which is the
    right shape for a spoken question.
    """

    name = "sarvam"

    def __init__(
        self,
        api_key: str,
        url: str = "https://api.sarvam.ai/speech-to-text",
        model: str = "saarika:v2.5",
        language_code: str = "unknown",
        timeout_s: float = 12.0,
    ):
        self.api_key = api_key
        self.url = url
        self.model = model
        self.language_code = language_code
        self.timeout_s = timeout_s

    async def transcribe(self, audio: bytes, filename: str, content_type: str) -> Transcript:
        data = {"model": self.model}
        if self.language_code:
            data["language_code"] = self.language_code

        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            resp = await client.post(
                self.url,
                headers={"api-subscription-key": self.api_key},
                files={"file": (filename, audio, content_type or "audio/wav")},
                data=data,
            )

        if resp.status_code >= 400:
            raise STTError(
                f"Sarvam STT {resp.status_code}: {resp.text[:300]}",
                status_code=resp.status_code,
            )

        payload = resp.json()
        text = payload.get("transcript") or payload.get("text") or ""
        return Transcript(
            text=text.strip(),
            language=payload.get("language_code") or payload.get("language"),
            provider=self.name,
            raw=payload,
        )


class ElevenLabsSTT(BaseSTT):
    name = "elevenlabs"

    def __init__(
        self,
        api_key: str,
        url: str = "https://api.elevenlabs.io/v1/speech-to-text",
        model: str = "scribe_v1",
        timeout_s: float = 12.0,
    ):
        self.api_key = api_key
        self.url = url
        self.model = model
        self.timeout_s = timeout_s

    async def transcribe(self, audio: bytes, filename: str, content_type: str) -> Transcript:
        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            resp = await client.post(
                self.url,
                headers={"xi-api-key": self.api_key},
                files={"file": (filename, audio, content_type or "audio/wav")},
                data={"model_id": self.model},
            )

        if resp.status_code >= 400:
            raise STTError(
                f"ElevenLabs STT {resp.status_code}: {resp.text[:300]}",
                status_code=resp.status_code,
            )

        payload = resp.json()
        return Transcript(
            text=(payload.get("text") or "").strip(),
            language=payload.get("language_code"),
            provider=self.name,
            raw=payload,
        )


def build_stt(settings) -> BaseSTT | None:
    provider = (settings.stt_provider or "none").lower()
    if provider == "sarvam":
        if not settings.sarvam_api_key:
            logger.warning("STT provider is 'sarvam' but SARVAM_API_KEY is unset; voice input disabled")
            return None
        return SarvamSTT(
            api_key=settings.sarvam_api_key,
            url=settings.sarvam_stt_url,
            model=settings.sarvam_model,
            language_code=settings.sarvam_language_code,
            timeout_s=settings.stt_timeout_s,
        )
    if provider == "elevenlabs":
        if not settings.elevenlabs_api_key:
            logger.warning("STT provider is 'elevenlabs' but ELEVENLABS_API_KEY is unset; voice input disabled")
            return None
        return ElevenLabsSTT(
            api_key=settings.elevenlabs_api_key,
            url=settings.elevenlabs_stt_url,
            model=settings.elevenlabs_model,
            timeout_s=settings.stt_timeout_s,
        )
    return None
