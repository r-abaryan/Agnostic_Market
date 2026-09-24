"""Loopback voice preview for the workbench, over the merchant's configured engines.

This is development inspection, not the voice pipeline. It reuses the same STT and TTS seams
the LiveKit worker builds from (`build_stt` / `build_tts`), so the merchant's pinned model,
brand voice, and keyterm bias are the ones exercised. What it does NOT exercise is everything
the pipeline owns around them: turn admission, VAD, barge-in, AI disclosure, and the durable
session boundary. A clean preview is not evidence that a call works.

The browser cannot hold a provider key, so capture and playback are proxied here instead.
"""

from __future__ import annotations

import io
import wave
from dataclasses import dataclass

from livekit import rtc
from livekit.agents.utils import http_context

from agnostic_market.dtos.config import VoiceConfig
from agnostic_market.dtos.llm import ProviderCredentialsConfig
from agnostic_market.secrets.base import SecretResolver
from agnostic_market.voice.stt_engine import build_stt
from agnostic_market.voice.tts_engine import build_tts

# Browsers capture at their own device rate; this is the rate the workbench resamples to before
# uploading, so both sides agree on one number rather than negotiating per request.
CAPTURE_SAMPLE_RATE = 16000
_MONO = 1
_BYTES_PER_SAMPLE = 2

# A preview turn is one short utterance. The ceiling bounds an accidental open microphone
# rather than defending a trust boundary: this API is loopback-only development state.
MAX_CAPTURE_SECONDS = 30
MAX_CAPTURE_BYTES = CAPTURE_SAMPLE_RATE * _BYTES_PER_SAMPLE * MAX_CAPTURE_SECONDS
MAX_SPEECH_CHARS = 2000


class VoicePreviewError(RuntimeError):
    """A preview request could not be served by the configured voice engines."""


@dataclass(frozen=True)
class SynthesizedSpeech:
    """One rendered utterance plus the identity of the voice that rendered it."""

    audio_wav: bytes
    provider: str
    model: str
    voice_id: str


def _wav_bytes(pcm: bytes, sample_rate: int, channels: int) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(_BYTES_PER_SAMPLE)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm)
    return buffer.getvalue()


class VoicePreview:
    """Synthesize and transcribe for one merchant, using that merchant's configured engines."""

    def __init__(
        self,
        voice: VoiceConfig,
        credentials: ProviderCredentialsConfig,
        secrets: SecretResolver,
    ) -> None:
        self._voice = voice
        self._credentials = credentials
        self._secrets = secrets

    @property
    def tts_identity(self) -> tuple[str, str, str]:
        config = self._voice.tts
        return (config.provider, config.model, config.voice_id)

    @property
    def stt_identity(self) -> tuple[str, str]:
        return (self._voice.stt.provider, self._voice.stt.model)

    async def synthesize(self, text: str) -> SynthesizedSpeech:
        spoken = text.strip()
        if not spoken:
            raise VoicePreviewError("nothing to speak")
        if len(spoken) > MAX_SPEECH_CHARS:
            raise VoicePreviewError("speech request exceeded the preview ceiling")

        chunks: list[bytes] = []
        sample_rate = 0
        channels = _MONO
        # The plugins take their aiohttp session from a job context only the LiveKit worker
        # binds. open() binds one here and passes through when a worker already has, so the
        # production seam is untouched. A session per request costs a handshake, which is the
        # right trade for a loopback preview serving one operator.
        async with http_context.open():
            engine = build_tts(self._voice.tts, self._credentials, self._secrets)
            try:
                stream = engine.synthesize(spoken)
                async for synthesized in stream:
                    frame = synthesized.frame
                    chunks.append(bytes(frame.data))
                    sample_rate = frame.sample_rate
                    channels = frame.num_channels
                await stream.aclose()
            except Exception as exc:  # provider transport, auth, or protocol failure
                raise VoicePreviewError(f"speech synthesis failed ({type(exc).__name__})") from exc
            finally:
                await engine.aclose()

        if not chunks or not sample_rate:
            raise VoicePreviewError("the speech engine returned no audio")
        provider, model, voice_id = self.tts_identity
        return SynthesizedSpeech(
            audio_wav=_wav_bytes(b"".join(chunks), sample_rate, channels),
            provider=provider,
            model=model,
            voice_id=voice_id,
        )

    async def transcribe(self, pcm: bytes) -> str:
        if not pcm:
            raise VoicePreviewError("no audio was captured")
        if len(pcm) > MAX_CAPTURE_BYTES:
            raise VoicePreviewError("captured audio exceeded the preview ceiling")
        if len(pcm) % _BYTES_PER_SAMPLE:
            raise VoicePreviewError("captured audio is not 16-bit PCM")

        frame = rtc.AudioFrame(
            data=pcm,
            sample_rate=CAPTURE_SAMPLE_RATE,
            num_channels=_MONO,
            samples_per_channel=len(pcm) // (_BYTES_PER_SAMPLE * _MONO),
        )
        async with http_context.open():
            engine = build_stt(self._voice.stt, self._credentials, self._secrets)
            try:
                event = await engine.recognize(frame)
            except Exception as exc:
                raise VoicePreviewError(f"transcription failed ({type(exc).__name__})") from exc
            finally:
                await engine.aclose()

        # Alternatives are ordered by confidence; an empty result means silence, not an error.
        alternatives = getattr(event, "alternatives", ()) or ()
        return str(alternatives[0].text).strip() if alternatives else ""
