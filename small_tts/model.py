"""Main Streaming TTS Model.

Combines Main Transformer + Depth Transformer with Mimi codec
for end-to-end streaming text-to-speech.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Generator, List
from dataclasses import dataclass

from small_tts.config import TTSConfig
from small_tts.transformer import MainTransformer, KVCache
from small_tts.depth import DepthTransformer
from small_tts.codec import MimiCodecWrapper, StreamingMimiDecoder


@dataclass
class StreamingState:
    """State for streaming inference."""
    kv_cache: KVCache
    prev_cb1_token: Optional[torch.Tensor] = None
    audio_buffer: Optional[torch.Tensor] = None
    text_buffer: List[int] = None
    frame_count: int = 0
    
    def __post_init__(self):
        if self.text_buffer is None:
            self.text_buffer = []


class StreamingTTS(nn.Module):
    """Streaming Text-to-Speech Model (~80M parameters).
    
    Architecture:
    - Main Transformer (70M): Predicts semantic codebook (CB1)
    - Depth Transformer (10M): Predicts acoustic codebooks (CB2-4)
    - Mimi Codec: Decodes tokens to audio
    
    Supports:
    - Input streaming: Text tokens arrive incrementally
    - Output streaming: Audio generated frame-by-frame
    - KV caching: Efficient autoregressive generation
    """
    
    def __init__(self, config: TTSConfig):
        super().__init__()
        
        self.config = config
        
        # Main temporal transformer for CB1 prediction
        self.main_transformer = MainTransformer(
            text_vocab_size=config.text_vocab_size,
            audio_vocab_size=config.audio_vocab_size,
            hidden_dim=config.main.hidden_dim,
            num_layers=config.main.num_layers,
            num_heads=config.main.num_heads,
            num_kv_heads=config.main.num_kv_heads,
            ffn_dim=config.main.ffn_dim,
            speaker_embed_dim=config.speaker_embed_dim,
            num_speakers=config.num_speakers,
            num_languages=config.num_languages,
            dropout=config.main.dropout,
            max_seq_len=config.main.max_seq_len,
        )
        
        # Depth transformer for CB2-4 prediction
        self.depth_transformer = DepthTransformer(
            audio_vocab_size=config.audio_vocab_size,
            num_predict_codebooks=config.num_codebooks - 1,  # CB2, CB3, CB4
            hidden_dim=config.depth.hidden_dim,
            num_layers=config.depth.num_layers,
            num_heads=config.depth.num_heads,
            ffn_dim=config.depth.ffn_dim,
            main_hidden_dim=config.main.hidden_dim,
            dropout=config.depth.dropout,
        )
        
        # Mimi codec for audio synthesis
        self.codec = MimiCodecWrapper(
            num_codebooks=config.num_codebooks,
            sample_rate=config.codec.sample_rate,
        )
        
    def forward(
        self,
        text_tokens: torch.Tensor,
        audio_tokens: torch.Tensor,
        speaker_id: Optional[torch.Tensor] = None,
        language_id: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass for training.
        
        Args:
            text_tokens: [batch, text_seq]
            audio_tokens: [batch, num_codebooks, audio_seq] - all 4 codebooks
            speaker_id: [batch]
            language_id: [batch]
            
        Returns:
            cb1_loss: Cross-entropy loss for CB1 prediction
            depth_loss: Cross-entropy loss for CB2-4 prediction
            total_loss: Combined loss
        """
        batch, num_cb, audio_seq = audio_tokens.shape
        
        # Extract codebooks
        cb1_tokens = audio_tokens[:, 0, :]  # [batch, seq]
        cb234_tokens = audio_tokens[:, 1:, :].permute(0, 2, 1)  # [batch, seq, 3]
        
        # Main transformer: predict CB1
        # For training, we use teacher forcing with CB1 shifted
        cb1_input = cb1_tokens[:, :-1]  # Input: all but last
        cb1_target = cb1_tokens[:, 1:]  # Target: all but first
        
        cb1_logits, hidden, _ = self.main_transformer(
            text_tokens=text_tokens,
            audio_tokens=cb1_input,
            speaker_id=speaker_id,
            language_id=language_id,
        )
        
        # Get hidden states corresponding to audio positions
        # Skip conditioning token and text tokens
        audio_start = 1 + text_tokens.shape[1]
        audio_hidden = hidden[:, audio_start:audio_start + cb1_input.shape[1], :]
        
        # CB1 loss
        cb1_logits_flat = cb1_logits[:, audio_start:audio_start + cb1_target.shape[1], :]
        cb1_loss = F.cross_entropy(
            cb1_logits_flat.reshape(-1, self.config.audio_vocab_size),
            cb1_target.reshape(-1),
            reduction="mean"
        )
        
        # Depth transformer: predict CB2-4
        cb234_target = cb234_tokens[:, 1:, :]  # Align with CB1 targets
        _, depth_loss = self.depth_transformer(
            cb1_tokens=cb1_target,  # Use ground truth CB1 for teacher forcing
            main_hidden=audio_hidden,
            target_tokens=cb234_target,
        )
        
        # Combined loss
        total_loss = cb1_loss + depth_loss
        
        return cb1_loss, depth_loss, total_loss
    
    def init_streaming(
        self,
        speaker_id: torch.Tensor,
        language_id: torch.Tensor,
        device: str = "cpu",
    ) -> StreamingState:
        """Initialize streaming state.
        
        Args:
            speaker_id: [batch] - speaker index
            language_id: [batch] - language index
            device: Device to use
            
        Returns:
            Streaming state
        """
        batch = speaker_id.shape[0]
        
        # Initialize empty KV cache
        kv_cache = KVCache.empty(self.main_transformer.num_layers)
        
        # Get conditioning and store in cache
        cond = self.main_transformer.get_conditioning(speaker_id, language_id)
        
        # Process conditioning through main transformer
        x = cond
        for i, layer in enumerate(self.main_transformer.layers):
            x, (k, v) = layer(x, kv_cache=None, start_pos=0)
            kv_cache.update(i, k, v)
        kv_cache.seq_len = 1
        
        return StreamingState(kv_cache=kv_cache)
    
    def stream_text(
        self,
        state: StreamingState,
        text_tokens: torch.Tensor,
    ) -> StreamingState:
        """Process incoming text tokens.
        
        Args:
            state: Current streaming state
            text_tokens: [batch, seq] - new text tokens
            
        Returns:
            Updated streaming state
        """
        # Embed and process text tokens
        text_emb = self.main_transformer.embed_text(text_tokens)
        
        x = text_emb
        for i, layer in enumerate(self.main_transformer.layers):
            layer_cache = state.kv_cache.get(i)
            x, (k, v) = layer(x, kv_cache=layer_cache, start_pos=state.kv_cache.seq_len)
            state.kv_cache.update(i, k, v)
            
        state.kv_cache.seq_len += text_tokens.shape[1]
        
        # Store text for later use
        state.text_buffer.extend(text_tokens[0].tolist())
        
        return state
    
    def generate_audio_frame(
        self,
        state: StreamingState,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> Tuple[torch.Tensor, StreamingState]:
        """Generate one audio frame (80ms).
        
        Args:
            state: Current streaming state
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling
            
        Returns:
            audio_chunk: [batch, 1, frame_size] - one frame of audio
            state: Updated streaming state
        """
        device = state.kv_cache.keys[0].device if state.kv_cache.keys[0] is not None else "cpu"
        batch = state.kv_cache.keys[0].shape[0] if state.kv_cache.keys[0] is not None else 1
        
        # Generate CB1 token
        if state.prev_cb1_token is None:
            # First audio frame - use a start token
            prev_token = torch.zeros(batch, 1, dtype=torch.long, device=device)
        else:
            prev_token = state.prev_cb1_token
            
        # Embed previous audio token
        audio_emb = self.main_transformer.embed_audio(prev_token)
        
        # Process through main transformer
        x = audio_emb
        for i, layer in enumerate(self.main_transformer.layers):
            layer_cache = state.kv_cache.get(i)
            x, (k, v) = layer(x, kv_cache=layer_cache, start_pos=state.kv_cache.seq_len)
            state.kv_cache.update(i, k, v)
            
        state.kv_cache.seq_len += 1
        
        # Get CB1 logits
        hidden = self.main_transformer.norm(x)
        logits = self.main_transformer.output_proj(hidden)
        
        # Sample CB1
        logits = logits[:, -1, :] / temperature
        
        if top_k > 0:
            indices_to_remove = logits < torch.topk(logits, top_k)[0][:, -1, None]
            logits[indices_to_remove] = float("-inf")
            
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
            sorted_indices_to_remove[:, 0] = False
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = float("-inf")
            
        probs = F.softmax(logits, dim=-1)
        cb1_token = torch.multinomial(probs, num_samples=1)  # [batch, 1]
        
        # Generate CB2-4 with depth transformer
        cb234_tokens = self.depth_transformer.generate(
            cb1_token=cb1_token,
            main_hidden=hidden,
            temperature=temperature,
        )  # [batch, 3]
        
        # Combine all codebooks
        all_tokens = torch.cat([cb1_token, cb234_tokens], dim=1)  # [batch, 4]
        all_tokens = all_tokens.unsqueeze(2)  # [batch, 4, 1]
        
        # Decode to audio
        audio_chunk = self.codec.decode(all_tokens)  # [batch, 1, frame_size]
        
        # Update state
        state.prev_cb1_token = cb1_token
        state.frame_count += 1
        
        return audio_chunk, state
    
    def generate_streaming(
        self,
        text_tokens: torch.Tensor,
        speaker_id: torch.Tensor,
        language_id: torch.Tensor,
        max_frames: int = 1000,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> Generator[torch.Tensor, None, None]:
        """Generate audio in streaming fashion.
        
        Yields audio chunks as they're generated.
        
        Args:
            text_tokens: [batch, text_seq]
            speaker_id: [batch]
            language_id: [batch]
            max_frames: Maximum number of frames to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling
            
        Yields:
            audio_chunk: [batch, 1, frame_size]
        """
        device = text_tokens.device
        
        # Initialize streaming state
        state = self.init_streaming(speaker_id, language_id, device)
        
        # Process text
        state = self.stream_text(state, text_tokens)
        
        # Generate audio frames
        for _ in range(max_frames):
            audio_chunk, state = self.generate_audio_frame(
                state, temperature, top_k, top_p
            )
            yield audio_chunk
            
            # TODO: Add end-of-speech detection
    
    def generate(
        self,
        text_tokens: torch.Tensor,
        speaker_id: torch.Tensor,
        language_id: torch.Tensor,
        max_frames: int = 1000,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> torch.Tensor:
        """Generate complete audio (non-streaming).
        
        Args:
            text_tokens: [batch, text_seq]
            speaker_id: [batch]
            language_id: [batch]
            max_frames: Maximum frames
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling
            
        Returns:
            audio: [batch, 1, num_samples]
        """
        audio_chunks = list(self.generate_streaming(
            text_tokens, speaker_id, language_id,
            max_frames, temperature, top_k, top_p
        ))
        
        return torch.cat(audio_chunks, dim=-1)
    
    def count_parameters(self) -> dict:
        """Count parameters in each component."""
        def count(module):
            return sum(p.numel() for p in module.parameters())
        
        return {
            "main_transformer": count(self.main_transformer),
            "depth_transformer": count(self.depth_transformer),
            "total": count(self),
        }


class StreamingTTSInference:
    """Inference wrapper for streaming TTS.
    
    Handles the full pipeline:
    1. Text tokenization
    2. Streaming generation
    3. Audio output
    """
    
    def __init__(
        self,
        model: StreamingTTS,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        
        # Simple character-level tokenizer for now
        # TODO: Replace with proper tokenizer
        self.char_to_id = {chr(i): i for i in range(256)}
        self.id_to_char = {i: chr(i) for i in range(256)}
        
    def tokenize(self, text: str) -> torch.Tensor:
        """Convert text to tokens."""
        tokens = [self.char_to_id.get(c, 0) for c in text]
        return torch.tensor([tokens], dtype=torch.long, device=self.device)
    
    @torch.no_grad()
    def synthesize(
        self,
        text: str,
        speaker_id: int = 0,
        language_id: int = 0,  # 0 = English, 1 = German
        max_duration: float = 30.0,
        temperature: float = 0.7,
    ) -> torch.Tensor:
        """Synthesize audio from text.
        
        Args:
            text: Input text
            speaker_id: Speaker index (0 or 1)
            language_id: Language (0=English, 1=German)
            max_duration: Maximum audio duration in seconds
            temperature: Sampling temperature
            
        Returns:
            audio: [samples] - audio waveform
        """
        text_tokens = self.tokenize(text)
        speaker = torch.tensor([speaker_id], device=self.device)
        language = torch.tensor([language_id], device=self.device)
        
        max_frames = int(max_duration * self.model.codec.frame_rate)
        
        audio = self.model.generate(
            text_tokens=text_tokens,
            speaker_id=speaker,
            language_id=language,
            max_frames=max_frames,
            temperature=temperature,
        )
        
        return audio[0, 0, :]  # [samples]
    
    @torch.no_grad()
    def synthesize_streaming(
        self,
        text: str,
        speaker_id: int = 0,
        language_id: int = 0,
        temperature: float = 0.7,
    ) -> Generator[torch.Tensor, None, None]:
        """Synthesize audio from text in streaming fashion.
        
        Yields audio chunks as they're generated (~80ms each).
        """
        text_tokens = self.tokenize(text)
        speaker = torch.tensor([speaker_id], device=self.device)
        language = torch.tensor([language_id], device=self.device)
        
        for audio_chunk in self.model.generate_streaming(
            text_tokens=text_tokens,
            speaker_id=speaker,
            language_id=language,
            temperature=temperature,
        ):
            yield audio_chunk[0, 0, :]


