# core/voice/whisper_engine.py

import asyncio
import logging
import tempfile
import numpy as np
import soundfile as sf
import os

logger = logging.getLogger("Sebastian.Whisper")

try:
    import whisper
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

DEFAULT_MODEL = "base"
DEFAULT_LANGUAGE = "en"

async def transcribe_audio_async(audio_data: np.ndarray, use_gpu: bool = False, model_size: str = DEFAULT_MODEL, language: str = DEFAULT_LANGUAGE) -> str:
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
            sf.write(tmpfile.name, audio_data, samplerate=16000)
            temp_path = tmpfile.name

        if TORCH_AVAILABLE:
            return await _transcribe_with_torch(temp_path, use_gpu, model_size, language)
        else:
            return await _transcribe_with_whisper_cpp(temp_path, language)

    except Exception as e:
        logger.exception("Failed to transcribe audio.")
        return ""
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


async def _transcribe_with_torch(file_path: str, use_gpu: bool, model_size: str, language: str) -> str:
    try:
        loop = asyncio.get_event_loop()
        model = await loop.run_in_executor(None, whisper.load_model, model_size)
        logger.info(f"Whisper model '{model_size}' loaded (Torch mode)")

        options = {"language": language, "fp16": use_gpu}
        result = await loop.run_in_executor(None, model.transcribe, file_path, options)
        return result.get("text", "").strip()

    except Exception as e:
        logger.exception("Torch Whisper transcription failed.")
        return ""


async def _transcribe_with_whisper_cpp(file_path: str, language: str) -> str:
    try:
        cmd = f"whisper-cpp --language {language} --model models/ggml-base.en.bin --file {file_path}"
        proc = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            logger.error(f"whisper.cpp failed: {stderr.decode().strip()}")
            return ""

        output = stdout.decode().strip()
        logger.info("whisper.cpp transcription complete.")
        return output

    except Exception as e:
        logger.exception("whisper.cpp transcription error.")
        return ""
