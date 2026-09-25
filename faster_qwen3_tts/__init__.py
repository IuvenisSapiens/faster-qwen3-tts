"""
faster-qwen3-tts: Real-time Qwen3-TTS inference with CUDA graphs or GGML
"""
from .ggml_backend import GGMLQwen3TTS
from .model import FasterQwen3TTS

__version__ = "0.5.2"
__all__ = ["FasterQwen3TTS", "GGMLQwen3TTS"]
