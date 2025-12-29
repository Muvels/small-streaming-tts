"""Core layers for Small Streaming TTS.

Implements efficient, portable building blocks:
- RMSNorm: Simpler than LayerNorm, easy C++ port
- RoPE: Rotary Position Embeddings for streaming
- SwiGLU: Better quality per parameter
- GQA: Grouped Query Attention for memory efficiency
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.
    
    Simpler and faster than LayerNorm - no mean computation.
    Easy to implement in C++.
    
    Formula: x * weight / sqrt(mean(x^2) + eps)
    """
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute RMS
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class RotaryEmbedding(nn.Module):
    """Rotary Position Embeddings (RoPE).
    
    Essential for streaming - no fixed sequence length.
    Encodes relative positions through rotation.
    """
    
    def __init__(self, dim: int, max_seq_len: int = 8192, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Precompute frequency bands
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # Cache for cos/sin values
        self._cos_cache = None
        self._sin_cache = None
        self._cache_seq_len = 0
        
    def _update_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        """Update the cos/sin cache if needed."""
        if seq_len <= self._cache_seq_len and self._cos_cache is not None:
            return
            
        self._cache_seq_len = max(seq_len, self.max_seq_len)
        
        # Compute positions
        t = torch.arange(self._cache_seq_len, device=device, dtype=dtype)
        
        # Compute frequencies
        freqs = torch.outer(t, self.inv_freq.to(device=device, dtype=dtype))
        
        # Duplicate for pairs (cos, sin applied to pairs of dims)
        emb = torch.cat([freqs, freqs], dim=-1)
        
        self._cos_cache = emb.cos().unsqueeze(0).unsqueeze(0)  # [1, 1, seq, dim]
        self._sin_cache = emb.sin().unsqueeze(0).unsqueeze(0)
        
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        start_pos: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary embeddings to query and key tensors.
        
        Args:
            q: Query tensor [batch, heads, seq, head_dim]
            k: Key tensor [batch, heads, seq, head_dim]
            start_pos: Starting position for streaming (for KV cache)
            
        Returns:
            Tuple of rotated (q, k)
        """
        seq_len = q.shape[2]
        self._update_cache(start_pos + seq_len, q.device, q.dtype)
        
        # Get relevant slice of cached values
        cos = self._cos_cache[:, :, start_pos:start_pos + seq_len, :self.dim]
        sin = self._sin_cache[:, :, start_pos:start_pos + seq_len, :self.dim]
        
        # Apply rotation
        q_rot = self._rotate(q, cos, sin)
        k_rot = self._rotate(k, cos, sin)
        
        return q_rot, k_rot
    
    def _rotate(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor
    ) -> torch.Tensor:
        """Apply rotation using cos and sin."""
        # Only apply to the portion that matches rope dimension
        rope_dim = min(self.dim, x.shape[-1])
        half_rope = rope_dim // 2
        
        # Split into pairs and rotate
        x1 = x[..., :half_rope]
        x2 = x[..., half_rope:rope_dim]
        
        # Get matching cos/sin slices
        cos_slice = cos[..., :half_rope]
        sin_slice = sin[..., :half_rope]
        
        # Rotate pairs
        rotated = torch.cat([
            x1 * cos_slice - x2 * sin_slice,
            x1 * sin_slice + x2 * cos_slice
        ], dim=-1)
        
        # If head_dim > rope_dim, concatenate the rest unchanged
        if x.shape[-1] > rope_dim:
            rotated = torch.cat([rotated, x[..., rope_dim:]], dim=-1)
            
        return rotated


class SwiGLU(nn.Module):
    """SwiGLU activation function with linear layers.
    
    Better quality than ReLU/GELU for same parameter count.
    Used in LLaMA and other modern transformers.
    
    Formula: (x @ W1) * silu(x @ W2) @ W3
    """
    
    def __init__(self, in_dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        # SwiGLU has 3 projections instead of 2
        self.w1 = nn.Linear(in_dim, hidden_dim, bias=False)  # Gate
        self.w2 = nn.Linear(in_dim, hidden_dim, bias=False)  # Up
        self.w3 = nn.Linear(hidden_dim, in_dim, bias=False)  # Down
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: (x @ W2) * silu(x @ W1) @ W3
        gate = F.silu(self.w1(x))
        up = self.w2(x)
        return self.dropout(self.w3(gate * up))


class GroupedQueryAttention(nn.Module):
    """Grouped Query Attention (GQA).
    
    Reduces KV cache by sharing key/value heads across query heads.
    With 12 query heads and 4 KV heads: 3x memory reduction.
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: Optional[int] = None,
        dropout: float = 0.0,
        max_seq_len: int = 8192,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_groups = num_heads // num_kv_heads
        self.head_dim = head_dim or (dim // num_heads)
        self.scale = self.head_dim ** -0.5
        
        # Projections
        self.q_proj = nn.Linear(dim, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * self.head_dim, dim, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        
        # RoPE
        self.rope = RotaryEmbedding(self.head_dim, max_seq_len)
        
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        start_pos: int = 0,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass with optional KV cache for streaming.
        
        Args:
            x: Input tensor [batch, seq, dim]
            mask: Attention mask [batch, 1, seq, seq] or None for causal
            kv_cache: Tuple of (cached_k, cached_v) or None
            start_pos: Starting position for RoPE (for streaming)
            
        Returns:
            output: [batch, seq, dim]
            new_kv_cache: Updated (k, v) cache
        """
        batch, seq_len, _ = x.shape
        
        # Project Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape to [batch, heads, seq, head_dim]
        q = q.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE
        q, k = self.rope(q, k, start_pos)
        
        # Handle KV cache
        if kv_cache is not None:
            cached_k, cached_v = kv_cache
            k = torch.cat([cached_k, k], dim=2)
            v = torch.cat([cached_v, v], dim=2)
            
        new_kv_cache = (k, v)
        
        # Expand KV heads to match query heads (GQA)
        if self.num_groups > 1:
            k = k.repeat_interleave(self.num_groups, dim=1)
            v = v.repeat_interleave(self.num_groups, dim=1)
        
        # Compute attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply causal mask if no mask provided
        if mask is None:
            # Causal mask for autoregressive generation
            kv_len = k.shape[2]
            causal_mask = torch.triu(
                torch.full((seq_len, kv_len), float("-inf"), device=x.device),
                diagonal=kv_len - seq_len + 1
            )
            attn_weights = attn_weights + causal_mask
        else:
            attn_weights = attn_weights + mask
            
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        output = torch.matmul(attn_weights, v)
        
        # Reshape and project output
        output = output.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        output = self.o_proj(output)
        
        return output, new_kv_cache


class TransformerBlock(nn.Module):
    """Single transformer block with GQA, SwiGLU, and RMSNorm.
    
    Pre-norm architecture for stability.
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
        max_seq_len: int = 8192,
    ):
        super().__init__()
        
        # Attention
        self.attn_norm = RMSNorm(dim)
        self.attn = GroupedQueryAttention(
            dim=dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            dropout=dropout,
            max_seq_len=max_seq_len,
        )
        
        # FFN
        self.ffn_norm = RMSNorm(dim)
        self.ffn = SwiGLU(dim, ffn_dim, dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        start_pos: int = 0,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass.
        
        Args:
            x: Input [batch, seq, dim]
            mask: Attention mask
            kv_cache: Optional KV cache for streaming
            start_pos: Position offset for RoPE
            
        Returns:
            output: [batch, seq, dim]
            new_kv_cache: Updated KV cache
        """
        # Attention with residual
        h = self.attn_norm(x)
        attn_out, new_kv_cache = self.attn(h, mask, kv_cache, start_pos)
        x = x + attn_out
        
        # FFN with residual
        h = self.ffn_norm(x)
        x = x + self.ffn(h)
        
        return x, new_kv_cache

