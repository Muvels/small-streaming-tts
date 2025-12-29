"""Small Streaming TTS with Mimi Codec.

A lightweight (~80M params) streaming text-to-speech model
using Kyutai's Mimi codec with 4 codebooks.

Architecture:
- Main Transformer (70M): Predicts semantic codebook (CB1)
- Depth Transformer (10M): Predicts acoustic codebooks (CB2-4)
- Streaming: Full input/output streaming with KV cache

Features:
- GQA (Grouped Query Attention) for memory efficiency
- RoPE for streaming-friendly position encoding
- SwiGLU activation for quality
- RMSNorm for simplicity
- Easy C++ export for deployment
"""

__version__ = "0.1.0"

from small_tts.config import TTSConfig
from small_tts.model import StreamingTTS, StreamingTTSInference
from small_tts.codec import MimiCodecWrapper
from small_tts.data import TextTokenizer

__all__ = [
    "TTSConfig",
    "StreamingTTS",
    "StreamingTTSInference",
    "MimiCodecWrapper",
    "TextTokenizer",
]
