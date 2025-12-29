"""Configuration for Small Streaming TTS."""

from dataclasses import dataclass, field
from typing import Optional
import yaml


@dataclass
class MainTransformerConfig:
    """Configuration for the main temporal transformer."""
    hidden_dim: int = 768  # 768 / 12 = 64 dim per head
    num_layers: int = 12
    num_heads: int = 12    # Must be divisible by num_kv_heads
    num_kv_heads: int = 4  # GQA: 3 query heads per KV head
    ffn_dim: int = 2048
    dropout: float = 0.1
    max_seq_len: int = 8192


@dataclass
class DepthTransformerConfig:
    """Configuration for the depth transformer."""
    hidden_dim: int = 512  # 512 / 8 = 64 dim per head
    num_layers: int = 4
    num_heads: int = 8
    ffn_dim: int = 1024
    dropout: float = 0.1


@dataclass
class CodecConfig:
    """Configuration for Mimi codec."""
    sample_rate: int = 24000
    frame_rate: float = 12.5
    num_codebooks: int = 4
    bandwidth: float = 1.1


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Stage 1
    stage1_epochs: int = 100
    stage1_batch_size: int = 32
    stage1_lr: float = 1e-4
    stage1_warmup_steps: int = 1000

    # Stage 2
    stage2_epochs: int = 50
    stage2_batch_size: int = 24
    stage2_lr: float = 5e-5
    stage2_warmup_steps: int = 500

    # Common
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    seed: int = 42
    num_workers: int = 4
    mixed_precision: bool = True
    gradient_accumulation: int = 1


@dataclass
class TTSConfig:
    """Main configuration for the TTS model."""
    # Model
    main: MainTransformerConfig = field(default_factory=MainTransformerConfig)
    depth: DepthTransformerConfig = field(default_factory=DepthTransformerConfig)
    codec: CodecConfig = field(default_factory=CodecConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    # Vocabulary
    text_vocab_size: int = 4096
    audio_vocab_size: int = 2048  # Per codebook
    num_codebooks: int = 4
    speaker_embed_dim: int = 256
    num_speakers: int = 2
    num_languages: int = 2

    # Paths
    train_path: str = "data/train"
    val_path: str = "data/val"
    log_dir: str = "logs"

    # Inference
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.9

    @classmethod
    def from_yaml(cls, path: str) -> "TTSConfig":
        """Load configuration from YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)

        config = cls()

        # Parse model config
        if "model" in data:
            model = data["model"]
            if "main" in model:
                for k, v in model["main"].items():
                    setattr(config.main, k, v)
            if "depth" in model:
                for k, v in model["depth"].items():
                    setattr(config.depth, k, v)

            # Top-level model params
            for key in ["text_vocab_size", "audio_vocab_size", "num_codebooks",
                       "speaker_embed_dim", "num_speakers", "num_languages"]:
                if key in model:
                    setattr(config, key, model[key])

        # Parse codec config
        if "codec" in data:
            for k, v in data["codec"].items():
                setattr(config.codec, k, v)

        # Parse training config
        if "training" in data:
            training = data["training"]
            if "stage1" in training:
                for k, v in training["stage1"].items():
                    setattr(config.training, f"stage1_{k}", v)
            if "stage2" in training:
                for k, v in training["stage2"].items():
                    setattr(config.training, f"stage2_{k}", v)
            for key in ["seed", "num_workers", "mixed_precision", "gradient_accumulation",
                       "weight_decay", "gradient_clip"]:
                if key in training:
                    setattr(config.training, key, training[key])

        # Parse paths
        if "data" in data:
            config.train_path = data["data"].get("train_path", config.train_path)
            config.val_path = data["data"].get("val_path", config.val_path)

        if "logging" in data:
            config.log_dir = data["logging"].get("log_dir", config.log_dir)

        if "inference" in data:
            for key in ["temperature", "top_k", "top_p"]:
                if key in data["inference"]:
                    setattr(config, key, data["inference"][key])

        return config

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            "main": self.main.__dict__,
            "depth": self.depth.__dict__,
            "codec": self.codec.__dict__,
            "training": self.training.__dict__,
            "text_vocab_size": self.text_vocab_size,
            "audio_vocab_size": self.audio_vocab_size,
            "num_codebooks": self.num_codebooks,
            "speaker_embed_dim": self.speaker_embed_dim,
            "num_speakers": self.num_speakers,
            "num_languages": self.num_languages,
        }


