from abc import abstractmethod
import io
import os
import torch
import torchaudio
import requests
from transformers import AutoModel
from openai import OpenAI
from typing import Optional
from deepgram import (
    DeepgramClient,
)


_ISO_CODE_TO_LANGUAGE = {
    "as": "assamese",
    "bn": "bengali",
    "brx": "bodo",
    "doi": "dogri",
    "gu": "gujarati",
    "hi": "hindi",
    "kn": "kannada",
    "ks": "kashmiri",
    "kok": "konkani",
    "ml": "malayalam",
    "mni": "manipuri",
    "mr": "marathi",
    "mai": "maithili",
    "ne": "nepali",
    "or": "odia",
    "pa": "punjabi",
    "sa": "sanskrit",
    "sat": "santali",
    "sd": "sindhi",
    "ta": "tamil",
    "te": "telugu",
    "ur": "urdu",
}


class BaseModel:
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
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModel.from_pretrained(
            "ai4bharat/indic-conformer-600m-multilingual", trust_remote_code=True
        ).to(self.device)

    @torch.inference_mode()
    def transcribe(self, audio: torch.Tensor, sampling_rate: int, language: Optional[str] = None) -> str:
        audio = audio.to(self.device)
        return self.model(wav=audio, lang=language, decoding="ctc")


class MenkaModel(BaseModel):
    def __init__(self):
        self.menka_url = os.environ.get("MENKA_URL", "http://0.0.0.0:8000")

    def transcribe(self, audio: torch.Tensor, sampling_rate: int, language: Optional[str] = None) -> str:
        buf = self.audio_to_wav_buffer(audio, sampling_rate)

        data = {"language": language} if language else {}
        r = requests.post(
            f"{self.menka_url}/transcribe",
            files={"file": ("audio.wav", buf, "audio/wav")},
            data=data,
        )
        r.raise_for_status()
        return r.json().get("text", "")


class GPT4oTranscribeModel(BaseModel):
    def __init__(self):
        assert os.environ.get("OPENAI_API_KEY"), "OPENAI_API_KEY is not set"
        self.client = OpenAI()

    @torch.inference_mode()
    def transcribe(self, audio: torch.Tensor, sampling_rate: int, language: Optional[str] = None) -> str:
        buffer = self.audio_to_wav_buffer(audio, sampling_rate)

        kwargs = {}
        # gpt4o-transcribe doesn't support language codes, so we need to prompt the model with language name to provide output in correct language.
        if language and language in _ISO_CODE_TO_LANGUAGE:
            kwargs["prompt"] = (
                f"The audio is in {_ISO_CODE_TO_LANGUAGE[language]}. Please transcribe in {_ISO_CODE_TO_LANGUAGE[language]}."
            )

        transcript = self.client.audio.transcriptions.create(
            model="gpt-4o-transcribe",
            file=buffer,
            **kwargs,
        )
        return transcript.text


class DeepgramNova3Model(BaseModel):
    def __init__(self):
        self.api_key = os.environ.get("DEEPGRAM_API_KEY")
        assert self.api_key, "DEEPGRAM_API_KEY is not set"

        self.client = DeepgramClient(api_key=self.api_key)
        self.model = "nova-3"

    @torch.inference_mode()
    def transcribe(self, audio: torch.Tensor, sampling_rate: int, language: Optional[str] = None) -> str:
        buf = self.audio_to_wav_buffer(audio, sampling_rate)

        response = self.client.listen.v1.media.transcribe_file(
            request=buf.getvalue(),
            model=self.model,
            language="multi",
            smart_format=True,
        )
        text = ""
        if response.results and response.results.channels and response.results.channels[0].alternatives:
            text = response.results.channels[0].alternatives[0].transcript
        return text
