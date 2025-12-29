"""Mimi Codec integration for Small Streaming TTS.

This module provides a wrapper around Kyutai's Mimi codec with:
- 4 codebook configuration (reduced from 8 for easier learning)
- Streaming encode/decode support
- Efficient token handling
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class MimiCodecWrapper(nn.Module):
    """Wrapper for Kyutai Mimi codec with 4 codebooks.
    
    Mimi processes 24 kHz audio at 12.5 Hz frame rate (80ms frames),
    producing 4 codebook tokens per frame for our configuration.
    
    The first codebook (CB1) captures semantic information (distilled from WavLM),
    while CB2-4 capture acoustic details progressively.
    """

    def __init__(
        self,
        num_codebooks: int = 4,
        sample_rate: int = 24000,
        device: str = "cpu",
    ):
        super().__init__()
        self.num_codebooks = num_codebooks
        self.sample_rate = sample_rate
        self.frame_rate = 12.5  # Hz
        self.frame_size = int(sample_rate / self.frame_rate)  # 1920 samples per frame
        self.device = device
        
        # Mimi codec vocabulary size per codebook
        self.vocab_size = 2048
        
        self._mimi = None
        self._initialized = False

    def _lazy_init(self):
        """Lazy initialization of Mimi codec."""
        if self._initialized:
            return
            
        try:
            from moshi.models import loaders
            
            # Load Mimi codec
            self._mimi = loaders.get_mimi(
                device=self.device,
                dtype=torch.float32,
            )
            self._mimi.set_num_codebooks(self.num_codebooks)
            self._initialized = True
            logger.info(f"Mimi codec initialized with {self.num_codebooks} codebooks")
            
        except ImportError as e:
            logger.warning(f"Mimi codec not available: {e}")
            logger.warning("Using mock codec for development")
            self._mimi = None
            self._initialized = True

    @property
    def mimi(self):
        """Get the Mimi codec, initializing if needed."""
        self._lazy_init()
        return self._mimi

    def encode(self, audio: torch.Tensor) -> torch.Tensor:
        """Encode audio to codec tokens.
        
        Args:
            audio: Audio waveform [batch, samples] or [batch, 1, samples]
            
        Returns:
            tokens: Codec tokens [batch, num_codebooks, num_frames]
        """
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)  # Add channel dim
            
        if self.mimi is not None:
            with torch.no_grad():
                tokens = self.mimi.encode(audio)
            # Take only first num_codebooks
            tokens = tokens[:, :self.num_codebooks, :]
        else:
            # Mock encoding for development
            batch_size = audio.shape[0]
            num_frames = audio.shape[-1] // self.frame_size
            tokens = torch.randint(
                0, self.vocab_size,
                (batch_size, self.num_codebooks, num_frames),
                device=audio.device
            )
            
        return tokens

    def decode(self, tokens: torch.Tensor) -> torch.Tensor:
        """Decode codec tokens to audio.
        
        Args:
            tokens: Codec tokens [batch, num_codebooks, num_frames]
            
        Returns:
            audio: Audio waveform [batch, 1, samples]
        """
        if self.mimi is not None:
            # Pad to 8 codebooks if Mimi requires it
            if tokens.shape[1] < 8:
                padding = torch.zeros(
                    tokens.shape[0],
                    8 - tokens.shape[1],
                    tokens.shape[2],
                    dtype=tokens.dtype,
                    device=tokens.device
                )
                tokens = torch.cat([tokens, padding], dim=1)
                
            with torch.no_grad():
                audio = self.mimi.decode(tokens)
        else:
            # Mock decoding for development
            batch_size = tokens.shape[0]
            num_frames = tokens.shape[-1]
            num_samples = num_frames * self.frame_size
            audio = torch.randn(batch_size, 1, num_samples, device=tokens.device)
            
        return audio

    def encode_streaming(self, audio_chunk: torch.Tensor) -> Optional[torch.Tensor]:
        """Encode a single audio chunk in streaming mode.
        
        Args:
            audio_chunk: Audio chunk [batch, samples] - should be frame_size samples
            
        Returns:
            tokens: Codec tokens [batch, num_codebooks, 1] or None if buffering
        """
        # For now, use regular encode
        # TODO: Implement proper streaming with Mimi's streaming API
        if audio_chunk.shape[-1] < self.frame_size:
            return None
            
        return self.encode(audio_chunk)

    def decode_streaming(self, tokens: torch.Tensor) -> torch.Tensor:
        """Decode a single frame of tokens in streaming mode.
        
        Args:
            tokens: Codec tokens [batch, num_codebooks, 1]
            
        Returns:
            audio: Audio chunk [batch, 1, frame_size]
        """
        return self.decode(tokens)

    def get_frame_size(self) -> int:
        """Get the number of audio samples per codec frame."""
        return self.frame_size

    def get_frame_rate(self) -> float:
        """Get the codec frame rate in Hz."""
        return self.frame_rate

    def tokens_to_audio_length(self, num_tokens: int) -> float:
        """Convert number of tokens to audio length in seconds."""
        return num_tokens / self.frame_rate

    def audio_length_to_tokens(self, seconds: float) -> int:
        """Convert audio length in seconds to number of tokens."""
        return int(seconds * self.frame_rate)


class StreamingMimiEncoder:
    """Streaming encoder for Mimi codec.
    
    Maintains internal buffer for streaming audio input.
    """
    
    def __init__(self, codec: MimiCodecWrapper):
        self.codec = codec
        self.buffer = None
        self.frame_size = codec.get_frame_size()
        
    def reset(self):
        """Reset the streaming buffer."""
        self.buffer = None
        
    def encode_chunk(self, audio: torch.Tensor) -> Optional[torch.Tensor]:
        """Encode an audio chunk, returning tokens when a full frame is available.
        
        Args:
            audio: Audio samples [batch, samples]
            
        Returns:
            tokens: [batch, num_codebooks, num_frames] or None if still buffering
        """
        if self.buffer is None:
            self.buffer = audio
        else:
            self.buffer = torch.cat([self.buffer, audio], dim=-1)
            
        # Check if we have enough samples for at least one frame
        if self.buffer.shape[-1] < self.frame_size:
            return None
            
        # Encode complete frames
        num_complete_frames = self.buffer.shape[-1] // self.frame_size
        samples_to_encode = num_complete_frames * self.frame_size
        
        audio_to_encode = self.buffer[..., :samples_to_encode]
        self.buffer = self.buffer[..., samples_to_encode:]
        
        return self.codec.encode(audio_to_encode)


class StreamingMimiDecoder:
    """Streaming decoder for Mimi codec.
    
    Decodes tokens frame by frame for streaming audio output.
    """
    
    def __init__(self, codec: MimiCodecWrapper):
        self.codec = codec
        
    def decode_frame(self, tokens: torch.Tensor) -> torch.Tensor:
        """Decode a single frame of tokens.
        
        Args:
            tokens: [batch, num_codebooks, 1]
            
        Returns:
            audio: [batch, 1, frame_size]
        """
        return self.codec.decode(tokens)
    
    def decode_frames(self, tokens: torch.Tensor) -> torch.Tensor:
        """Decode multiple frames of tokens.
        
        Args:
            tokens: [batch, num_codebooks, num_frames]
            
        Returns:
            audio: [batch, 1, num_frames * frame_size]
        """
        return self.codec.decode(tokens)


