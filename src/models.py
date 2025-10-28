from abc import abstractmethod
import io
import os
import threading
import torch
import torchaudio
import requests
from transformers import AutoModel
from openai import OpenAI
from typing import Optional
from deepgram import (
    DeepgramClient,
)
import pycountry
from sarvamai import SarvamAI
from google import genai
from google.genai import types


class BaseModel:
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def transcribe(self, audio: torch.Tensor, sampling_rate: int, language: Optional[str] = None) -> str:
        pass

    def audio_to_wav_buffer(self, audio: torch.Tensor, sampling_rate: int) -> io.BytesIO:
        buf = io.BytesIO()
        buf.name = "audio.wav"
        torchaudio.save(buf, audio, sampling_rate, format="wav")
        buf.seek(0)
        return buf


class IndicConformerModel(BaseModel):
    def __init__(self, name: str):
        super().__init__(name)
        assert os.environ.get("HF_TOKEN"), "HF_TOKEN is not set"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModel.from_pretrained(
            "ai4bharat/indic-conformer-600m-multilingual", trust_remote_code=True
        ).to(self.device)
        # Taken from https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual/blob/main/assets/language_masks.json
        self.supported_languages = {
            "as",
            "bn",
            "brx",
            "doi",
            "kok",
            "gu",
            "hi",
            "kn",
            "ks",
            "mai",
            "ml",
            "mr",
            "mni",
            "ne",
            "or",
            "pa",
            "sa",
            "sat",
            "sd",
            "ta",
            "te",
            "ur",
        }

    @torch.inference_mode()
    def transcribe(self, audio: torch.Tensor, sampling_rate: int, language: Optional[str] = None) -> str:
        audio = audio.to(self.device)
        try:
            text = self.model(wav=audio, lang=language, decoding="ctc")
            return text
        except Exception as e:
            print(f"Error during IndicConformer transcription: {e}")
        return None


class MenkaModel(BaseModel):
    def __init__(self, name: str):
        super().__init__(name)
        self.menka_base_url = os.environ.get("MENKA_BASE_URL", "http://0.0.0.0")
        self.menka_urls = [f"{self.menka_base_url}:{port}" for port in range(8000, 8008)]
        self.supported_languages = {
            "as",
            "bn",
            "brx",
            "doi",
            "kok",
            "gu",
            "hi",
            "kn",
            "ks",
            "mai",
            "ml",
            "mr",
            "mni",
            "ne",
            "or",
            "pa",
            "sa",
            "sat",
            "sd",
            "ta",
            "te",
            "ur",
            "en",
        }
        self._rr_lock = threading.Lock()
        self._rr_index = 0

    def _next_menka_url(self) -> str:
        with self._rr_lock:
            url = self.menka_urls[self._rr_index]
            self._rr_index = (self._rr_index + 1) % len(self.menka_urls)
        return url

    def transcribe(self, audio: torch.Tensor, sampling_rate: int, language: Optional[str] = None) -> str:
        buf = self.audio_to_wav_buffer(audio, sampling_rate)

        data = {"language": language} if language else {}
        try:
            menka_url = self._next_menka_url()
            r = requests.post(
                f"{menka_url}/transcribe",
                files={"file": ("audio.wav", buf, "audio/wav")},
                data=data,
            )
            r.raise_for_status()
            return r.json().get("text")
        except Exception as e:
            print(f"Error during Menka transcription: {e}")
        return None


class GPT4oTranscribeModel(BaseModel):
    def __init__(self, name: str):
        super().__init__(name)
        assert os.environ.get("OPENAI_API_KEY"), "OPENAI_API_KEY is not set"
        self.client = OpenAI()
        # Taken from https://platform.openai.com/docs/guides/speech-to-text/supported-languages#supported-languages
        self.supported_languages = {
            "af",
            "ar",
            "hy",
            "az",
            "be",
            "bs",
            "bg",
            "ca",
            "zh",
            "hr",
            "cs",
            "da",
            "nl",
            "en",
            "et",
            "fi",
            "fr",
            "gl",
            "de",
            "el",
            "he",
            "hi",
            "hu",
            "is",
            "id",
            "it",
            "ja",
            "kn",
            "kk",
            "ko",
            "lv",
            "lt",
            "mk",
            "ms",
            "mr",
            "mi",
            "ne",
            "no",
            "fa",
            "pl",
            "pt",
            "ro",
            "ru",
            "sr",
            "sk",
            "sl",
            "es",
            "sw",
            "sv",
            "tl",
            "ta",
            "th",
            "tr",
            "uk",
            "ur",
            "vi",
            "cy",
        }

    def transcribe(self, audio: torch.Tensor, sampling_rate: int, language: Optional[str] = None) -> str:
        buffer = self.audio_to_wav_buffer(audio, sampling_rate)

        kwargs = {}
        if language:
            if language in self.supported_languages:
                kwargs["language"] = language
            else:
                lang = None
                if len(language) == 2:
                    lang = pycountry.languages.get(alpha_2=language)
                elif len(language) == 3:
                    lang = pycountry.languages.get(alpha_3=language)
                if lang:
                    kwargs["prompt"] = f"Output the transcript in {lang.name}."

        try:
            transcript = self.client.audio.transcriptions.create(
                model="gpt-4o-transcribe",
                file=buffer,
                **kwargs,
            )
            return transcript.text
        except Exception as e:
            print(f"Error during GPT4o transcription: {e}")
        return None


class DeepgramNova3Model(BaseModel):
    def __init__(self, name: str):
        super().__init__(name)
        self.api_key = os.environ.get("DEEPGRAM_API_KEY")
        assert self.api_key, "DEEPGRAM_API_KEY is not set"

        self.client = DeepgramClient(api_key=self.api_key)
        self.model = "nova-3"
        # Taken from https://developers.deepgram.com/docs/models-languages-overview
        self.supported_languages = {"en", "es", "fr", "de", "hi", "ru", "pt", "ja", "it", "nl"}

    def transcribe(self, audio: torch.Tensor, sampling_rate: int, language: Optional[str] = None) -> str:
        if language and language not in self.supported_languages:
            return None

        buf = self.audio_to_wav_buffer(audio, sampling_rate)

        try:
            response = self.client.listen.v1.media.transcribe_file(
                request=buf.getvalue(),
                model=self.model,
                language="multi",
                smart_format=True,
            )
            text = response.results.channels[0].alternatives[0].transcript
            return text
        except Exception as e:
            print(f"Error during Deepgram transcription: {e}")
        return None


class SarvamAIModel(BaseModel):
    def __init__(self, name: str):
        super().__init__(name)
        self.api_key = os.environ.get("SARVAM_API_KEY")
        assert self.api_key, "SARVAM_API_KEY is not set"

        self.client = SarvamAI(api_subscription_key=self.api_key)
        # Taken from https://docs.sarvam.ai/api-reference-docs/getting-started/models#common-supported-languages
        self.supported_languages = {"hi", "bn", "kn", "ml", "mr", "od", "pa", "ta", "te", "en", "gu"}

    def transcribe(self, audio: torch.Tensor, sampling_rate: int, language: Optional[str] = None) -> str:
        if language and language not in self.supported_languages:
            language = "unknown"
        elif language in self.supported_languages:
            language += "-IN"

        buf = self.audio_to_wav_buffer(audio, sampling_rate)

        try:
            response = self.client.speech_to_text.transcribe(file=buf, model="saarika:v2.5", language_code=language)
            return response.transcript
        except Exception as e:
            print(f"Error during SarvamAI transcription: {e}")
        return None

class GeminiModel(BaseModel):
    def __init__(self, name: str):
        super().__init__(name)
        self.api_key = os.environ.get("GEMINI_API_KEY")
        assert self.api_key, "GEMINI_API_KEY is not set"

        self.client = genai.Client(api_key=self.api_key)
        self.model = "gemini-2.5-pro"
        self.supported_languages = {
            "as",
            "bn",
            "brx",
            "doi",
            "kok",
            "gu",
            "hi",
            "kn",
            "ks",
            "mai",
            "ml",
            "mr",
            "mni",
            "ne",
            "or",
            "pa",
            "sa",
            "sat",
            "sd",
            "ta",
            "te",
            "ur",
            "en",
        }
        
        self.config = types.GenerateContentConfig(
            system_instruction=(
                "You are a transcription engine. Detect the spoken language and "
                "return the transcript in the ORIGINAL language using its native script. "
                "Do NOT translate"
                "Only return the transcript text (no extra commentary)."
            )
        )

    def transcribe(self, audio: torch.Tensor, sampling_rate: int, language: Optional[str] = None) -> str:
        buf = self.audio_to_wav_buffer(audio, sampling_rate)
        try:
            resp = self.client.models.generate_content(
                model=self.model,
                contents=[
                    types.Part.from_bytes(data=buf.getvalue(), mime_type="audio/wav"),
                ],
                config=self.config,
            )
            return resp.text
        except Exception as e:
            print(f"Error during Gemini transcription: {e}")
        return None