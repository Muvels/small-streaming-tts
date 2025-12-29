# Small Streaming TTS

A lightweight (~80M parameters) streaming text-to-speech model using Kyutai's Mimi codec.

## Features

- **Small & Fast**: ~80M parameters, runs real-time on CPU
- **Streaming**: Full input and output streaming support
- **Multi-speaker**: Supports 2 speakers (male/female)
- **Bilingual**: German and English
- **Modern Architecture**: GQA, RoPE, SwiGLU, RMSNorm
- **Easy Deployment**: Binary export format for C++ inference

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Streaming TTS Model                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Text Tokens ─┬──► Main Transformer (70M) ──► CB1 Token     │
│               │    - 12 layers                               │
│  Speaker ID  ─┤    - 768 hidden dim                         │
│               │    - GQA (12 heads, 4 KV)                    │
│  Language ID ─┘    - RoPE + SwiGLU + RMSNorm                │
│                                                              │
│                           │                                  │
│                           ▼                                  │
│                                                              │
│  CB1 + Hidden ──► Depth Transformer (10M) ──► CB2,CB3,CB4   │
│                   - 4 layers                                 │
│                   - 512 hidden dim                          │
│                   - Parallel prediction                      │
│                                                              │
│                           │                                  │
│                           ▼                                  │
│                                                              │
│  [CB1,CB2,CB3,CB4] ──► Mimi Decoder ──► Audio Stream        │
│                        (Kyutai)          (24kHz, 80ms/frame) │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/small-tts.git
cd small-tts

# Install dependencies
pip install -e .

# Or install requirements directly
pip install -r requirements.txt
```

## Quick Start

```python
from small_tts import TTSConfig, StreamingTTS, StreamingTTSInference

# Create model
config = TTSConfig()
model = StreamingTTS(config)

# Load trained weights
# model.load_state_dict(torch.load("model.pt")["model_state_dict"])

# Create inference wrapper
inference = StreamingTTSInference(model, device="cpu")

# Generate audio
audio = inference.synthesize(
    text="Hello, world!",
    speaker_id=0,  # 0=female, 1=male
    language_id=0, # 0=English, 1=German
)

# Or stream audio
for chunk in inference.synthesize_streaming(text="Hello, world!"):
    # Process 80ms audio chunk
    play_audio(chunk)
```

## Training

### Prepare Data

Organize your data as:
```
data/
├── train/
│   ├── manifest.json
│   └── audio/
│       ├── 001.wav
│       ├── 002.wav
│       └── ...
└── val/
    ├── manifest.json
    └── audio/
        └── ...
```

`manifest.json` format:
```json
[
  {
    "audio": "audio/001.wav",
    "text": "Hello world",
    "speaker": 0,
    "language": 0,
    "duration": 2.5
  }
]
```

### Run Training

```bash
# Two-stage training
python -m small_tts.train \
    --config config/default.yaml \
    --data_dir data/train \
    --val_dir data/val

# Resume from checkpoint
python -m small_tts.train \
    --config config/default.yaml \
    --resume logs/checkpoints/stage1_final.pt
```

### Training Stages

1. **Stage 1**: Train main transformer for CB1 (semantic) prediction
   - 100 epochs
   - Teacher forcing with ground truth CB1

2. **Stage 2**: Joint training with depth transformer
   - 50 epochs
   - Train CB2-4 prediction alongside CB1

## Export for C++

```bash
# Export to binary format
python -m small_tts.export \
    --checkpoint logs/checkpoints/final.pt \
    --output_dir export/ \
    --format binary \
    --quantize \
    --generate_header
```

This produces:
- `model.bin` or `model_fp16.bin`: Model weights
- `config.json`: Model configuration
- `small_tts.h`: C++ header for loading

## Configuration

Key configuration options in `config/default.yaml`:

```yaml
model:
  main:
    hidden_dim: 768      # Main transformer hidden size
    num_layers: 12       # Number of transformer layers
    num_heads: 12        # Attention heads
    num_kv_heads: 4      # KV heads for GQA
    ffn_dim: 2048        # FFN intermediate size
  
  depth:
    hidden_dim: 512      # Depth transformer hidden size
    num_layers: 4        # Depth transformer layers
  
  num_codebooks: 4       # Mimi codebooks to use

training:
  stage1:
    epochs: 100
    learning_rate: 1.0e-4
  stage2:
    epochs: 50
    learning_rate: 5.0e-5
```

## Model Details

### Main Transformer (~70M params)

- **Attention**: Grouped Query Attention (GQA) with 12 query heads and 4 KV heads
- **Position**: Rotary Position Embeddings (RoPE) - no maximum length
- **Activation**: SwiGLU - better quality per parameter
- **Normalization**: RMSNorm - faster, simpler
- **Streaming**: Full KV cache support for token-by-token generation

### Depth Transformer (~10M params)

- Predicts CB2, CB3, CB4 in parallel given CB1
- Cross-attends to main transformer hidden state
- Runs once per audio frame (~80ms)

### Mimi Codec

- 24kHz audio, 12.5Hz frame rate
- 4 codebooks (reduced from 8 for easier learning)
- CB1: Semantic information (distilled from WavLM)
- CB2-4: Acoustic details

## Streaming Inference

The model supports true bi-directional streaming:

1. **Input Streaming**: Text tokens can arrive incrementally
2. **Output Streaming**: Audio is generated frame-by-frame (80ms chunks)
3. **KV Caching**: Previous computations are cached for efficiency

Latency budget per frame:
- Main transformer: ~5-10ms (CPU)
- Depth transformer: ~3-5ms
- Mimi decode: ~5-10ms
- **Total**: ~95-105ms per 80ms frame

## License

[Add your license here]

## Acknowledgments

- [Kyutai](https://github.com/kyutai-labs/moshi) for the Mimi codec
- Architecture inspired by Moshi and modern LLM research


