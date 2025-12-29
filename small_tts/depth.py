"""Depth Transformer for Small Streaming TTS.

Predicts acoustic codebooks (CB2-4) from semantic codebook (CB1)
and the main transformer's hidden state.

~10M parameters, runs once per audio frame.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from small_tts.layers import RMSNorm, SwiGLU, GroupedQueryAttention


class DepthAttention(nn.Module):
    """Attention for depth transformer.
    
    Non-causal within the codebook dimension since we have CB1.
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Query input [batch, seq, dim]
            context: Key/value context [batch, ctx_seq, dim], optional
            
        Returns:
            output: [batch, seq, dim]
        """
        batch, seq_len, _ = x.shape
        
        q = self.q_proj(x)
        
        # Self-attention or cross-attention
        kv_input = context if context is not None else x
        k = self.k_proj(kv_input)
        v = self.v_proj(kv_input)
        
        # Reshape
        q = q.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        
        return self.o_proj(out)


class DepthBlock(nn.Module):
    """Single depth transformer block."""
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        ffn_dim: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # Self-attention
        self.self_attn_norm = RMSNorm(dim)
        self.self_attn = DepthAttention(dim, num_heads, dropout)
        
        # Cross-attention to main transformer
        self.cross_attn_norm = RMSNorm(dim)
        self.cross_attn = DepthAttention(dim, num_heads, dropout)
        
        # FFN
        self.ffn_norm = RMSNorm(dim)
        self.ffn = SwiGLU(dim, ffn_dim, dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Codebook embeddings [batch, num_codebooks, dim]
            context: Main transformer hidden state [batch, 1, context_dim]
            
        Returns:
            output: [batch, num_codebooks, dim]
        """
        # Self-attention over codebooks
        h = self.self_attn_norm(x)
        x = x + self.self_attn(h)
        
        # Cross-attention to main transformer
        h = self.cross_attn_norm(x)
        x = x + self.cross_attn(h, context)
        
        # FFN
        h = self.ffn_norm(x)
        x = x + self.ffn(h)
        
        return x


class DepthTransformer(nn.Module):
    """Depth Transformer for predicting CB2-4 (~10M parameters).
    
    Architecture:
    - 4 layers
    - 512 hidden dimension
    - 8 attention heads
    - SwiGLU FFN with 1024 intermediate
    
    Takes CB1 embedding + main transformer hidden state,
    predicts CB2, CB3, CB4 in parallel (one forward pass).
    """
    
    def __init__(
        self,
        # Vocabulary
        audio_vocab_size: int = 2048,
        num_predict_codebooks: int = 3,  # CB2, CB3, CB4
        # Model dims
        hidden_dim: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        ffn_dim: int = 1024,
        # Input from main transformer
        main_hidden_dim: int = 768,
        # Other
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_predict_codebooks = num_predict_codebooks
        self.audio_vocab_size = audio_vocab_size
        
        # CB1 embedding (input)
        self.cb1_embed = nn.Embedding(audio_vocab_size, hidden_dim)
        
        # Codebook position embeddings (for CB2, CB3, CB4)
        self.codebook_pos_embed = nn.Parameter(
            torch.zeros(1, num_predict_codebooks, hidden_dim)
        )
        
        # Project main transformer hidden state
        self.context_proj = nn.Linear(main_hidden_dim, hidden_dim)
        
        # Depth transformer layers
        self.layers = nn.ModuleList([
            DepthBlock(
                dim=hidden_dim,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        
        # Output heads for each codebook
        self.norm = RMSNorm(hidden_dim)
        self.output_heads = nn.ModuleList([
            nn.Linear(hidden_dim, audio_vocab_size, bias=False)
            for _ in range(num_predict_codebooks)
        ])
        
        # Initialize
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                
        nn.init.zeros_(self.codebook_pos_embed)
        
    def forward(
        self,
        cb1_tokens: torch.Tensor,
        main_hidden: torch.Tensor,
        target_tokens: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass for training or inference.
        
        Args:
            cb1_tokens: CB1 tokens [batch, seq] or [batch, 1] for single frame
            main_hidden: Hidden state from main transformer [batch, seq, main_dim]
            target_tokens: Target CB2-4 tokens [batch, seq, 3] for training
            
        Returns:
            logits: [batch, seq, 3, vocab_size] - predictions for CB2, CB3, CB4
            loss: Cross-entropy loss if targets provided
        """
        batch, seq_len = cb1_tokens.shape
        
        # Embed CB1
        cb1_emb = self.cb1_embed(cb1_tokens)  # [batch, seq, dim]
        
        # Project main hidden state
        context = self.context_proj(main_hidden)  # [batch, seq, dim]
        
        # Create input for each position: CB1 embedding + position embedding
        # We'll predict CB2, CB3, CB4 in parallel
        # Shape: [batch * seq, num_predict, dim]
        x = cb1_emb.unsqueeze(2).expand(-1, -1, self.num_predict_codebooks, -1)
        x = x + self.codebook_pos_embed  # Add codebook position
        
        # Reshape for batch processing
        x = x.view(batch * seq_len, self.num_predict_codebooks, self.hidden_dim)
        context = context.view(batch * seq_len, 1, self.hidden_dim)
        
        # Forward through layers
        for layer in self.layers:
            x = layer(x, context)
            
        # Output projections
        x = self.norm(x)  # [batch * seq, 3, dim]
        x = x.view(batch, seq_len, self.num_predict_codebooks, self.hidden_dim)
        
        # Get logits for each codebook
        logits_list = []
        for i, head in enumerate(self.output_heads):
            logits_i = head(x[:, :, i, :])  # [batch, seq, vocab]
            logits_list.append(logits_i)
            
        logits = torch.stack(logits_list, dim=2)  # [batch, seq, 3, vocab]
        
        # Compute loss if targets provided
        loss = None
        if target_tokens is not None:
            # target_tokens: [batch, seq, 3]
            loss = F.cross_entropy(
                logits.reshape(-1, self.audio_vocab_size),
                target_tokens.reshape(-1),
                reduction="mean"
            )
            
        return logits, loss
    
    def generate(
        self,
        cb1_token: torch.Tensor,
        main_hidden: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Generate CB2-4 tokens for a single frame.
        
        Args:
            cb1_token: CB1 token [batch, 1]
            main_hidden: Hidden from main transformer [batch, 1, main_dim]
            temperature: Sampling temperature
            
        Returns:
            tokens: [batch, 3] - CB2, CB3, CB4 tokens
        """
        logits, _ = self.forward(cb1_token, main_hidden)
        # logits: [batch, 1, 3, vocab]
        
        logits = logits[:, 0, :, :] / temperature  # [batch, 3, vocab]
        
        # Sample from each codebook
        tokens = []
        for i in range(self.num_predict_codebooks):
            probs = F.softmax(logits[:, i, :], dim=-1)
            token = torch.multinomial(probs, num_samples=1)
            tokens.append(token)
            
        return torch.cat(tokens, dim=1)  # [batch, 3]


class ParallelDepthTransformer(DepthTransformer):
    """Parallel version of DepthTransformer.
    
    Predicts all codebooks in a single forward pass using
    separate output heads. This is faster than the mini-AR
    variant but may have slightly lower quality.
    """
    pass  # Same as DepthTransformer - parallel by default


class MiniARDepthTransformer(nn.Module):
    """Mini-Autoregressive Depth Transformer.
    
    Predicts codebooks sequentially: CB1 → CB2 → CB3 → CB4
    Each step conditions on previous codebooks.
    
    Slightly higher quality but 3x more steps per frame.
    """
    
    def __init__(
        self,
        audio_vocab_size: int = 2048,
        num_predict_codebooks: int = 3,
        hidden_dim: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        ffn_dim: int = 1024,
        main_hidden_dim: int = 768,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_predict_codebooks = num_predict_codebooks
        self.audio_vocab_size = audio_vocab_size
        
        # Embeddings for all codebooks
        self.codebook_embeds = nn.ModuleList([
            nn.Embedding(audio_vocab_size, hidden_dim)
            for _ in range(num_predict_codebooks + 1)  # CB1 + CB2, CB3, CB4
        ])
        
        # Context projection
        self.context_proj = nn.Linear(main_hidden_dim, hidden_dim)
        
        # Single transformer that processes sequentially
        self.layers = nn.ModuleList([
            DepthBlock(
                dim=hidden_dim,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        
        # Single output head (shared across codebooks)
        self.norm = RMSNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, audio_vocab_size, bias=False)
        
    def forward_step(
        self,
        prev_tokens: torch.Tensor,
        codebook_idx: int,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """Forward one step in the depth dimension.
        
        Args:
            prev_tokens: Previous codebook tokens [batch]
            codebook_idx: Which codebook we're predicting (0=CB2, 1=CB3, 2=CB4)
            context: Main transformer hidden [batch, 1, dim]
            
        Returns:
            logits: [batch, vocab]
        """
        # Embed previous token
        x = self.codebook_embeds[codebook_idx](prev_tokens)  # [batch, dim]
        x = x.unsqueeze(1)  # [batch, 1, dim]
        
        # Forward through layers
        for layer in self.layers:
            x = layer(x, context)
            
        x = self.norm(x[:, 0, :])
        return self.output_proj(x)
    
    def generate(
        self,
        cb1_token: torch.Tensor,
        main_hidden: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Generate CB2-4 tokens sequentially.
        
        Args:
            cb1_token: CB1 token [batch, 1]
            main_hidden: Hidden from main transformer [batch, 1, main_dim]
            temperature: Sampling temperature
            
        Returns:
            tokens: [batch, 3] - CB2, CB3, CB4 tokens
        """
        batch = cb1_token.shape[0]
        context = self.context_proj(main_hidden)
        
        prev_token = cb1_token.squeeze(1)  # [batch]
        tokens = []
        
        for i in range(self.num_predict_codebooks):
            logits = self.forward_step(prev_token, i, context)
            logits = logits / temperature
            
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
            tokens.append(next_token)
            prev_token = next_token
            
        return torch.stack(tokens, dim=1)  # [batch, 3]

