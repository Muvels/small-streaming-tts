"""Main Temporal Transformer for Small Streaming TTS.

This is the core model that predicts semantic tokens (CB1) from text input.
~70M parameters, designed for streaming with KV cache.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from dataclasses import dataclass

from small_tts.layers import RMSNorm, TransformerBlock


@dataclass
class KVCache:
    """Key-Value cache for streaming inference."""
    keys: List[torch.Tensor]
    values: List[torch.Tensor]
    seq_len: int = 0
    
    @classmethod
    def empty(cls, num_layers: int) -> "KVCache":
        """Create empty cache."""
        return cls(
            keys=[None] * num_layers,
            values=[None] * num_layers,
            seq_len=0
        )
    
    def update(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor):
        """Update cache for a layer."""
        self.keys[layer_idx] = k
        self.values[layer_idx] = v
        
    def get(self, layer_idx: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Get cached KV for a layer."""
        k, v = self.keys[layer_idx], self.values[layer_idx]
        if k is None:
            return None
        return (k, v)


class MainTransformer(nn.Module):
    """Main Temporal Transformer (~70M parameters).
    
    Architecture:
    - 12 layers
    - 768 hidden dimension
    - 12 attention heads with 4 KV heads (GQA)
    - SwiGLU FFN with 2048 intermediate
    - RoPE position encoding
    - RMSNorm
    
    This model predicts the semantic codebook (CB1) autoregressively.
    """
    
    def __init__(
        self,
        # Vocabulary
        text_vocab_size: int = 4096,
        audio_vocab_size: int = 2048,
        # Model dims
        hidden_dim: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        num_kv_heads: int = 4,
        ffn_dim: int = 2048,
        # Conditioning
        speaker_embed_dim: int = 256,
        num_speakers: int = 2,
        num_languages: int = 2,
        # Other
        dropout: float = 0.1,
        max_seq_len: int = 8192,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.text_vocab_size = text_vocab_size
        self.audio_vocab_size = audio_vocab_size
        
        # Token embeddings
        self.text_embed = nn.Embedding(text_vocab_size, hidden_dim)
        self.audio_embed = nn.Embedding(audio_vocab_size, hidden_dim)
        
        # Special tokens
        self.text_type_embed = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.audio_type_embed = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        
        # Speaker and language conditioning
        self.speaker_embed = nn.Embedding(num_speakers, speaker_embed_dim)
        self.language_embed = nn.Embedding(num_languages, speaker_embed_dim)
        self.cond_proj = nn.Linear(speaker_embed_dim * 2, hidden_dim)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(
                dim=hidden_dim,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                ffn_dim=ffn_dim,
                dropout=dropout,
                max_seq_len=max_seq_len,
            )
            for _ in range(num_layers)
        ])
        
        # Output
        self.norm = RMSNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, audio_vocab_size, bias=False)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights for stability."""
        # Standard initialization
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                
        # Scale output projection for stability
        nn.init.normal_(self.output_proj.weight, mean=0.0, std=0.02 / (2 * self.num_layers) ** 0.5)
        
    def get_conditioning(
        self,
        speaker_id: torch.Tensor,
        language_id: torch.Tensor,
    ) -> torch.Tensor:
        """Get speaker/language conditioning embedding.
        
        Args:
            speaker_id: [batch]
            language_id: [batch]
            
        Returns:
            conditioning: [batch, 1, hidden_dim]
        """
        speaker = self.speaker_embed(speaker_id)  # [batch, speaker_dim]
        language = self.language_embed(language_id)  # [batch, speaker_dim]
        
        cond = torch.cat([speaker, language], dim=-1)  # [batch, speaker_dim * 2]
        cond = self.cond_proj(cond)  # [batch, hidden_dim]
        
        return cond.unsqueeze(1)  # [batch, 1, hidden_dim]
        
    def embed_text(self, text_tokens: torch.Tensor) -> torch.Tensor:
        """Embed text tokens.
        
        Args:
            text_tokens: [batch, seq]
            
        Returns:
            embeddings: [batch, seq, hidden_dim]
        """
        return self.text_embed(text_tokens) + self.text_type_embed
    
    def embed_audio(self, audio_tokens: torch.Tensor) -> torch.Tensor:
        """Embed audio tokens (CB1).
        
        Args:
            audio_tokens: [batch, seq]
            
        Returns:
            embeddings: [batch, seq, hidden_dim]
        """
        return self.audio_embed(audio_tokens) + self.audio_type_embed
        
    def forward(
        self,
        text_tokens: torch.Tensor,
        audio_tokens: Optional[torch.Tensor] = None,
        speaker_id: Optional[torch.Tensor] = None,
        language_id: Optional[torch.Tensor] = None,
        text_lengths: Optional[torch.Tensor] = None,
        audio_lengths: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[KVCache]]:
        """Forward pass for training or inference.
        
        For training: provide text_tokens and audio_tokens (teacher forcing)
        For inference: provide text_tokens only, use kv_cache for streaming
        
        Args:
            text_tokens: [batch, text_seq]
            audio_tokens: [batch, audio_seq] - CB1 tokens for teacher forcing
            speaker_id: [batch] - speaker index
            language_id: [batch] - language index
            kv_cache: KV cache for streaming
            
        Returns:
            logits: [batch, seq, audio_vocab_size] - CB1 predictions
            hidden: [batch, seq, hidden_dim] - for depth transformer
            new_cache: Updated KV cache
        """
        batch_size = text_tokens.shape[0]
        device = text_tokens.device
        
        # Default speaker/language
        if speaker_id is None:
            speaker_id = torch.zeros(batch_size, dtype=torch.long, device=device)
        if language_id is None:
            language_id = torch.zeros(batch_size, dtype=torch.long, device=device)
            
        # Get conditioning
        cond = self.get_conditioning(speaker_id, language_id)
        
        # Embed tokens
        text_emb = self.embed_text(text_tokens)
        
        if audio_tokens is not None:
            audio_emb = self.embed_audio(audio_tokens)
            # Interleave: [cond, text, audio] - simplified version
            # In practice, you'd want more sophisticated interleaving
            x = torch.cat([cond, text_emb, audio_emb], dim=1)
        else:
            x = torch.cat([cond, text_emb], dim=1)

        # Optional key padding mask (training-time only).
        # This prevents attention to padded TEXT/AUDIO tokens, which otherwise makes the model
        # learn batch-padding artifacts and hurts conditioning quality.
        attn_mask = None
        if kv_cache is None and (text_lengths is not None or audio_lengths is not None):
            bsz = x.shape[0]
            cond_len = 1
            text_len = text_tokens.shape[1]
            audio_len = audio_tokens.shape[1] if audio_tokens is not None else 0
            total_len = cond_len + text_len + audio_len

            key_valid = torch.ones(bsz, total_len, dtype=torch.bool, device=x.device)

            if text_lengths is not None:
                tl = text_lengths.to(device=x.device).clamp(min=0, max=text_len)
                tpos = torch.arange(text_len, device=x.device)[None, :]
                text_valid = tpos < tl[:, None]
                key_valid[:, cond_len:cond_len + text_len] = text_valid

            if audio_tokens is not None and audio_lengths is not None:
                # audio_tokens here are the *input* CB1 tokens (shifted), so valid input length is (audio_lengths - 1)
                al = (audio_lengths.to(device=x.device) - 1).clamp(min=0, max=audio_len)
                apos = torch.arange(audio_len, device=x.device)[None, :]
                audio_valid = apos < al[:, None]
                key_valid[:, cond_len + text_len:cond_len + text_len + audio_len] = audio_valid

            # SDPA boolean mask: True = attend allowed, False = masked out.
            # Shape chosen to broadcast across heads and query length.
            attn_mask = key_valid[:, None, None, :]  # [B, 1, 1, S]
            
        # Determine start position for RoPE
        start_pos = 0
        if kv_cache is not None and kv_cache.seq_len > 0:
            start_pos = kv_cache.seq_len
            
        # Create new cache if needed
        new_cache = kv_cache if kv_cache is not None else KVCache.empty(self.num_layers)
        
        # Forward through layers
        for i, layer in enumerate(self.layers):
            layer_cache = new_cache.get(i)
            x, (k, v) = layer(x, mask=attn_mask, kv_cache=layer_cache, start_pos=start_pos)
            new_cache.update(i, k, v)
            
        new_cache.seq_len = start_pos + x.shape[1]
        
        # Output projection
        hidden = self.norm(x)
        logits = self.output_proj(hidden)
        
        return logits, hidden, new_cache
    
    def generate_step(
        self,
        text_tokens: torch.Tensor,
        prev_audio_token: Optional[torch.Tensor],
        speaker_id: torch.Tensor,
        language_id: torch.Tensor,
        kv_cache: Optional[KVCache],
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> Tuple[torch.Tensor, torch.Tensor, KVCache]:
        """Generate a single audio token (streaming step).
        
        Args:
            text_tokens: [batch, text_seq] - new text tokens
            prev_audio_token: [batch, 1] - previous CB1 token
            speaker_id: [batch]
            language_id: [batch]
            kv_cache: Cached KV from previous steps
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling
            
        Returns:
            next_token: [batch, 1] - sampled CB1 token
            hidden: [batch, 1, hidden_dim] - for depth transformer
            new_cache: Updated cache
        """
        # Embed current input
        if kv_cache is None or kv_cache.seq_len == 0:
            # First step - include conditioning
            cond = self.get_conditioning(speaker_id, language_id)
            text_emb = self.embed_text(text_tokens)
            x = torch.cat([cond, text_emb], dim=1)
        else:
            # Subsequent steps - just the new token(s)
            if prev_audio_token is not None:
                x = self.embed_audio(prev_audio_token)
            else:
                x = self.embed_text(text_tokens)
                
        start_pos = kv_cache.seq_len if kv_cache is not None else 0
        new_cache = kv_cache if kv_cache is not None else KVCache.empty(self.num_layers)
        
        # Forward through layers
        for i, layer in enumerate(self.layers):
            layer_cache = new_cache.get(i)
            x, (k, v) = layer(x, kv_cache=layer_cache, start_pos=start_pos)
            new_cache.update(i, k, v)
            
        new_cache.seq_len = start_pos + x.shape[1]
        
        # Get last position logits
        hidden = self.norm(x[:, -1:, :])
        logits = self.output_proj(hidden)  # [batch, 1, vocab]
        
        # Sample
        logits = logits[:, -1, :] / temperature  # [batch, vocab]
        
        # Top-k filtering
        if top_k > 0:
            indices_to_remove = logits < torch.topk(logits, top_k)[0][:, -1, None]
            logits[indices_to_remove] = float("-inf")
            
        # Top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
            sorted_indices_to_remove[:, 0] = False
            
            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove
            )
            logits[indices_to_remove] = float("-inf")
            
        # Sample from distribution
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        return next_token, hidden, new_cache


