"""Mimi Codec integration for Small Streaming TTS.

This module provides a wrapper around Kyutai's Mimi codec with:
- 4 codebook configuration (reduced from 8 for easier learning)
- Streaming encode/decode support
- Efficient token handling

Uses Hugging Face Transformers implementation of Mimi.
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
    
    Uses Hugging Face Transformers implementation.
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
        self._feature_extractor = None
        self._initialized = False

    def _lazy_init(self, input_device: str = None):
        """Lazy initialization of Mimi codec from Hugging Face.
        
        Args:
            input_device: Device of input tensor (used to determine where to run codec)
        """
        if self._initialized:
            return
        
        # Determine the device to use
        # If input_device is provided and different from init device, update
        if input_device is not None:
            self.device = input_device
            
        try:
            from transformers import MimiModel, AutoFeatureExtractor

            # Suppress very noisy NNPACK warnings on some CPU environments.
            # (NNPACK is an optional CPU acceleration backend; failing to init is not fatal.)
            try:
                import torch.backends.nnpack as nnpack  # type: ignore
                nnpack.set_flags(_enabled=False)
            except Exception:
                pass
            
            print("Loading Mimi codec from Hugging Face (kyutai/mimi)...")
            
            # Mimi works best on CPU; MPS has compatibility issues
            codec_device = "cpu"  # Always use CPU for Mimi for compatibility
            
            # Load the pre-trained Mimi model
            self._mimi = MimiModel.from_pretrained("kyutai/mimi")
            self._mimi = self._mimi.to(codec_device)
            self._mimi.eval()
            self._codec_device = codec_device
            
            # Note: num_quantizers is the total (32), we only USE first num_codebooks
            # Don't change the config, just use first N codebooks
            
            # Load the feature extractor
            self._feature_extractor = AutoFeatureExtractor.from_pretrained("kyutai/mimi")
            
            self._initialized = True
            print(f"Mimi codec loaded! Using {self.num_codebooks} codebooks, decoder on {codec_device}")
            
        except ImportError as e:
            print(f"WARNING: Hugging Face Transformers Mimi not available: {e}")
            print("Using mock codec - output will be random noise!")
            self._mimi = None
            self._codec_device = "cpu"
            self._initialized = True
        except Exception as e:
            print(f"WARNING: Failed to load Mimi codec: {e}")
            print("Using mock codec - output will be random noise!")
            self._mimi = None
            self._codec_device = "cpu"
            self._initialized = True

    def _get_mimi(self, input_device: str = None):
        """Get the Mimi codec, initializing if needed."""
        self._lazy_init(input_device)
        return self._mimi
    
    @property
    def mimi(self):
        """Get the Mimi codec (property for backward compatibility)."""
        self._lazy_init()
        return self._mimi
    
    @property
    def feature_extractor(self):
        """Get the feature extractor, initializing if needed."""
        self._lazy_init()
        return self._feature_extractor

    def encode(self, audio: torch.Tensor) -> torch.Tensor:
        """Encode audio to codec tokens.
        
        Args:
            audio: Audio waveform [batch, samples] or [batch, 1, samples]
            
        Returns:
            tokens: Codec tokens [batch, num_codebooks, num_frames]
        """
        original_device = audio.device
        
        # Ensure [batch, channels, samples] format
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)  # Add channel dim
        
        # Initialize with input device context
        mimi = self._get_mimi(str(original_device))
            
        if mimi is not None:
            with torch.no_grad():
                # Move to codec device (CPU for Mimi compatibility)
                audio = audio.to(self._codec_device)
                
                # Create padding mask (all ones = no padding)
                padding_mask = torch.ones(audio.shape[0], audio.shape[2], dtype=torch.long, device=self._codec_device)
                
                # Encode using HuggingFace Mimi
                # IMPORTANT: request exactly num_codebooks quantizers so encode+decode is consistent.
                encoder_outputs = mimi.encode(audio, padding_mask=padding_mask, num_quantizers=self.num_codebooks)
                tokens = encoder_outputs.audio_codes  # [batch, num_codebooks, num_frames]
                
                # Move back to original device
                tokens = tokens.to(original_device)
        else:
            # Mock encoding for development
            batch_size = audio.shape[0]
            num_frames = audio.shape[-1] // self.frame_size
            tokens = torch.randint(
                0, self.vocab_size,
                (batch_size, self.num_codebooks, num_frames),
                device=original_device
            )
            
        return tokens

    def decode(self, tokens: torch.Tensor) -> torch.Tensor:
        """Decode codec tokens to audio.
        
        Args:
            tokens: Codec tokens [batch, num_codebooks, num_frames]
            
        Returns:
            audio: Audio waveform [batch, 1, samples]
        """
        original_device = tokens.device
        
        # Initialize with input device context
        mimi = self._get_mimi(str(original_device))
        
        if mimi is not None:
            with torch.no_grad():
                # Move to codec device (CPU for Mimi compatibility)
                tokens = tokens.to(self._codec_device).long()

                # IMPORTANT:
                # - Do NOT pad to 32 quantizers. If the codes were produced with 4 codebooks,
                #   padding the remaining 28 with zeros can yield garbage/noise.
                # - Use MimiModel.decode(), which correctly reconstructs waveform for the
                #   provided number of quantizers.
                decoder_out = mimi.decode(audio_codes=tokens)

                # transformers returns MimiDecoderOutput with .audio_values
                audio = decoder_out.audio_values if hasattr(decoder_out, "audio_values") else decoder_out[0]

                # Ensure output is [batch, 1, samples]
                if audio.dim() == 2:
                    audio = audio.unsqueeze(1)

                # Move back to original device
                audio = audio.to(original_device)
        else:
            # Mock decoding for development
            batch_size = tokens.shape[0]
            num_frames = tokens.shape[-1]
            num_samples = num_frames * self.frame_size
            audio = torch.randn(batch_size, 1, num_samples, device=original_device)
            
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


