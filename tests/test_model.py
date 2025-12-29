"""Tests for Small Streaming TTS model."""

import torch
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from small_tts.config import TTSConfig
from small_tts.model import StreamingTTS
from small_tts.layers import RMSNorm, SwiGLU, GroupedQueryAttention, RotaryEmbedding
from small_tts.transformer import MainTransformer, KVCache
from small_tts.depth import DepthTransformer


class TestLayers:
    """Test individual layers."""
    
    def test_rmsnorm(self):
        """Test RMSNorm layer."""
        norm = RMSNorm(768)
        x = torch.randn(2, 10, 768)
        y = norm(x)
        
        assert y.shape == x.shape
        # Check that output is normalized
        rms = torch.sqrt(torch.mean(y ** 2, dim=-1))
        assert torch.allclose(rms, torch.ones_like(rms), atol=0.1)
    
    def test_swiglu(self):
        """Test SwiGLU activation."""
        ffn = SwiGLU(768, 2048)
        x = torch.randn(2, 10, 768)
        y = ffn(x)
        
        assert y.shape == x.shape
    
    def test_rope(self):
        """Test Rotary Position Embeddings."""
        rope = RotaryEmbedding(64, max_seq_len=1024)
        
        q = torch.randn(2, 12, 10, 64)
        k = torch.randn(2, 12, 10, 64)
        
        q_rot, k_rot = rope(q, k, start_pos=0)
        
        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape
        
        # Test streaming (different start positions)
        q_rot2, k_rot2 = rope(q, k, start_pos=100)
        
        # Rotations should be different for different positions
        assert not torch.allclose(q_rot, q_rot2)
    
    def test_gqa(self):
        """Test Grouped Query Attention."""
        attn = GroupedQueryAttention(
            dim=768,
            num_heads=12,
            num_kv_heads=4,
        )
        
        x = torch.randn(2, 10, 768)
        y, kv_cache = attn(x)
        
        assert y.shape == x.shape
        assert kv_cache[0].shape[1] == 4  # 4 KV heads
        
    def test_gqa_with_cache(self):
        """Test GQA with KV cache for streaming."""
        attn = GroupedQueryAttention(
            dim=768,
            num_heads=12,
            num_kv_heads=4,
        )
        
        # First pass
        x1 = torch.randn(2, 5, 768)
        y1, cache1 = attn(x1, start_pos=0)
        
        # Second pass with cache
        x2 = torch.randn(2, 3, 768)
        y2, cache2 = attn(x2, kv_cache=cache1, start_pos=5)
        
        assert y2.shape == (2, 3, 768)
        assert cache2[0].shape[2] == 8  # 5 + 3 cached positions


class TestMainTransformer:
    """Test Main Transformer."""
    
    def test_forward(self):
        """Test forward pass."""
        model = MainTransformer(
            text_vocab_size=4096,
            audio_vocab_size=2048,
            hidden_dim=256,  # Smaller for testing
            num_layers=2,
            num_heads=4,
            num_kv_heads=2,
            ffn_dim=512,
        )
        
        text_tokens = torch.randint(0, 4096, (2, 10))
        audio_tokens = torch.randint(0, 2048, (2, 20))
        speaker_id = torch.zeros(2, dtype=torch.long)
        language_id = torch.zeros(2, dtype=torch.long)
        
        logits, hidden, cache = model(
            text_tokens=text_tokens,
            audio_tokens=audio_tokens,
            speaker_id=speaker_id,
            language_id=language_id,
        )
        
        # 1 (cond) + 10 (text) + 20 (audio) = 31
        expected_seq_len = 1 + 10 + 20
        assert logits.shape == (2, expected_seq_len, 2048)
        assert hidden.shape == (2, expected_seq_len, 256)
    
    def test_streaming(self):
        """Test streaming generation with KV cache."""
        model = MainTransformer(
            text_vocab_size=4096,
            audio_vocab_size=2048,
            hidden_dim=256,
            num_layers=2,
            num_heads=4,
            num_kv_heads=2,
            ffn_dim=512,
        )
        
        text_tokens = torch.randint(0, 4096, (1, 10))
        speaker_id = torch.zeros(1, dtype=torch.long)
        language_id = torch.zeros(1, dtype=torch.long)
        
        # Generate step by step
        token, hidden, cache = model.generate_step(
            text_tokens=text_tokens,
            prev_audio_token=None,
            speaker_id=speaker_id,
            language_id=language_id,
            kv_cache=None,
        )
        
        assert token.shape == (1, 1)
        assert cache.seq_len > 0
        initial_seq_len = cache.seq_len
        
        # Continue generation with a fresh text token
        new_text = torch.randint(0, 4096, (1, 1))
        token2, hidden2, cache2 = model.generate_step(
            text_tokens=new_text,
            prev_audio_token=token,
            speaker_id=speaker_id,
            language_id=language_id,
            kv_cache=cache,
        )
        
        assert token2.shape == (1, 1)
        # Cache should grow by 1 (the new audio token)
        assert cache2.seq_len == initial_seq_len + 1


class TestDepthTransformer:
    """Test Depth Transformer."""
    
    def test_forward(self):
        """Test forward pass."""
        model = DepthTransformer(
            audio_vocab_size=2048,
            num_predict_codebooks=3,
            hidden_dim=256,
            num_layers=2,
            num_heads=4,
            ffn_dim=512,
            main_hidden_dim=256,
        )
        
        cb1_tokens = torch.randint(0, 2048, (2, 10))
        main_hidden = torch.randn(2, 10, 256)
        
        logits, loss = model(cb1_tokens, main_hidden)
        
        assert logits.shape == (2, 10, 3, 2048)
        assert loss is None
    
    def test_with_targets(self):
        """Test forward with targets for training."""
        model = DepthTransformer(
            audio_vocab_size=2048,
            num_predict_codebooks=3,
            hidden_dim=256,
            num_layers=2,
            main_hidden_dim=256,
        )
        
        cb1_tokens = torch.randint(0, 2048, (2, 10))
        main_hidden = torch.randn(2, 10, 256)
        targets = torch.randint(0, 2048, (2, 10, 3))
        
        logits, loss = model(cb1_tokens, main_hidden, target_tokens=targets)
        
        assert loss is not None
        assert loss.item() > 0
    
    def test_generate(self):
        """Test single-frame generation."""
        model = DepthTransformer(
            audio_vocab_size=2048,
            num_predict_codebooks=3,
            hidden_dim=256,
            num_layers=2,
            main_hidden_dim=256,
        )
        
        cb1_token = torch.randint(0, 2048, (2, 1))
        main_hidden = torch.randn(2, 1, 256)
        
        tokens = model.generate(cb1_token, main_hidden)
        
        assert tokens.shape == (2, 3)  # CB2, CB3, CB4


class TestStreamingTTS:
    """Test full StreamingTTS model."""
    
    def test_creation(self):
        """Test model creation and parameter counting."""
        config = TTSConfig()
        # Use smaller dims for testing (must be divisible by heads)
        config.main.hidden_dim = 256
        config.main.num_layers = 2
        config.main.num_heads = 4
        config.main.num_kv_heads = 2
        config.main.ffn_dim = 512
        config.depth.hidden_dim = 128
        config.depth.num_layers = 2
        config.depth.num_heads = 4
        
        model = StreamingTTS(config)
        params = model.count_parameters()
        
        assert "main_transformer" in params
        assert "depth_transformer" in params
        assert params["total"] > 0
    
    def test_forward(self):
        """Test training forward pass."""
        config = TTSConfig()
        config.main.hidden_dim = 256
        config.main.num_layers = 2
        config.main.num_heads = 4
        config.main.num_kv_heads = 2
        config.main.ffn_dim = 512
        config.depth.hidden_dim = 128
        config.depth.num_layers = 2
        config.depth.num_heads = 4
        
        model = StreamingTTS(config)
        
        text_tokens = torch.randint(0, 4096, (2, 10))
        audio_tokens = torch.randint(0, 2048, (2, 4, 20))  # [batch, codebooks, seq]
        speaker_id = torch.zeros(2, dtype=torch.long)
        language_id = torch.zeros(2, dtype=torch.long)
        
        cb1_loss, depth_loss, total_loss = model(
            text_tokens, audio_tokens, speaker_id, language_id
        )
        
        assert cb1_loss.item() > 0
        assert depth_loss.item() > 0
        assert total_loss.item() > 0
    
    def test_streaming_generation(self):
        """Test streaming audio generation."""
        config = TTSConfig()
        config.main.hidden_dim = 256
        config.main.num_layers = 2
        config.main.num_heads = 4
        config.main.num_kv_heads = 2
        config.main.ffn_dim = 512
        config.depth.hidden_dim = 128
        config.depth.num_layers = 2
        config.depth.num_heads = 4
        
        model = StreamingTTS(config)
        model.eval()
        
        text_tokens = torch.randint(0, 4096, (1, 10))
        speaker_id = torch.zeros(1, dtype=torch.long)
        language_id = torch.zeros(1, dtype=torch.long)
        
        # Generate a few frames
        frames = []
        for i, chunk in enumerate(model.generate_streaming(
            text_tokens, speaker_id, language_id,
            max_frames=5,
        )):
            frames.append(chunk)
            if i >= 4:
                break
        
        assert len(frames) == 5
        for frame in frames:
            assert frame.shape[1] == 1  # Single channel


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

