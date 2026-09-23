"""
faster-qwen3-tts: Real-time Qwen3-TTS inference using CUDA graphs
"""
from .ggml_backend import GGMLQwen3TTS
from .model import FasterQwen3TTS

__version__ = "0.4.0"
__all__ = ["FasterQwen3TTS", "GGMLQwen3TTS"]
