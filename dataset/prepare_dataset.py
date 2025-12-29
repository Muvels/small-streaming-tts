"""
Dataset preparation script for Small Streaming TTS.

Converts podcast-style dataset to training format:
- Parses transcript files
- Encodes audio with Mimi codec to get tokens (4 codebooks)
- Generates JSONL manifest

Input format:
    test_dataset/
    └── de/
        └── <conversation_id>/
            ├── full_conversation.wav
            └── segments/
                ├── 001_speaker1.wav
                ├── 002_speaker2.wav
                └── vibevoice-podcast-script.txt

Output format:
    output/
    ├── train.jsonl
    ├── val.jsonl
    └── tokens/
        ├── de_conv1_001.pt
        ├── de_conv1_002.pt
        └── ...

Note: Uses 4 codebooks (reduced from Mimi's default 8) for easier learning
in small TTS models.
"""

import os
import re
import json
import argparse
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from tqdm import tqdm

import torch
import numpy as np

# Try multiple audio backends
def load_audio(audio_path: str) -> tuple:
    """Load audio file, returning (waveform, sample_rate)."""
    try:
        import soundfile as sf
        data, sr = sf.read(audio_path)
        # Convert to torch tensor [channels, samples]
        if data.ndim == 1:
            waveform = torch.from_numpy(data).float().unsqueeze(0)
        else:
            waveform = torch.from_numpy(data.T).float()
        return waveform, sr
    except ImportError:
        pass
    
    try:
        from scipy.io import wavfile
        sr, data = wavfile.read(audio_path)
        # Normalize to float
        if data.dtype == np.int16:
            data = data.astype(np.float32) / 32768.0
        elif data.dtype == np.int32:
            data = data.astype(np.float32) / 2147483648.0
        # Convert to torch tensor [channels, samples]
        if data.ndim == 1:
            waveform = torch.from_numpy(data).float().unsqueeze(0)
        else:
            waveform = torch.from_numpy(data.T).float()
        return waveform, sr
    except ImportError:
        pass
    
    try:
        import torchaudio
        return torchaudio.load(audio_path)
    except:
        pass
    
    raise ImportError("No audio backend available. Install soundfile: pip install soundfile")


@dataclass
class Sample:
    """Single training sample."""
    text: str
    speaker_id: int  # 0 or 1
    audio_path: str
    tokens_path: str
    language: str
    conversation_id: str
    segment_idx: int


def parse_transcript(transcript_path: str) -> List[Tuple[int, str]]:
    """
    Parse transcript file.
    
    Format:
        [1]: Hey, hast du letztens die neue Aufgaben-App ausprobiert?
        [2]: Ja, total handy...
    
    Returns:
        List of (speaker_num, text) tuples
    """
    lines = []
    pattern = re.compile(r'\[(\d+)\]:\s*(.+)')
    
    with open(transcript_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            match = pattern.match(line)
            if match:
                speaker_num = int(match.group(1))
                text = match.group(2).strip()
                lines.append((speaker_num, text))
    
    return lines


def find_audio_for_segment(
    segments_dir: Path,
    segment_idx: int,
) -> Optional[Path]:
    """
    Find audio file for a segment index.
    
    Matches patterns like:
        001_speaker1.wav
        001_speaker2.wav
        segment_001.wav
    """
    # Try common naming patterns
    patterns = [
        f"{segment_idx:03d}_speaker*.wav",
        f"{segment_idx:03d}_*.wav",
        f"segment_{segment_idx:03d}.wav",
        f"{segment_idx:03d}.wav",
    ]
    
    for pattern in patterns:
        matches = list(segments_dir.glob(pattern))
        if matches:
            return matches[0]
    
    return None


NUM_CODEBOOKS = 4  # Use 4 codebooks for small TTS (reduced from 8)


def load_mimi_codec(device: str = "cpu", num_codebooks: int = NUM_CODEBOOKS):
    """
    Load Mimi codec for audio tokenization.
    
    Uses Kyutai's moshi package with reduced codebooks for smaller model.
    
    Args:
        device: Device to load model on
        num_codebooks: Number of codebooks to use (default: 4)
    
    Returns:
        (model, codec_type) tuple
    """
    try:
        # Try kyutai's moshi package (preferred)
        from moshi.models import loaders
        mimi = loaders.get_mimi(device=device)
        mimi.set_num_codebooks(num_codebooks)  # Reduce to 4 codebooks
        mimi.eval()
        print(f"Loaded Mimi with {num_codebooks} codebooks")
        return mimi, "moshi"
    except ImportError as e:
        print(f"Could not import moshi: {e}")
    except Exception as e:
        print(f"Error loading moshi Mimi: {e}")
    
    try:
        # Try huggingface transformers as fallback
        from transformers import MimiModel, AutoFeatureExtractor
        model = MimiModel.from_pretrained("kyutai/mimi")
        model = model.to(device)
        model.eval()
        feature_extractor = AutoFeatureExtractor.from_pretrained("kyutai/mimi")
        print(f"Loaded HuggingFace Mimi (will use {num_codebooks} codebooks)")
        return (model, feature_extractor, num_codebooks), "transformers"
    except ImportError as e:
        print(f"Could not import transformers: {e}")
    except Exception as e:
        print(f"Error loading HuggingFace Mimi: {e}")
    
    return None, None


def resample_audio(waveform: torch.Tensor, sr: int, target_sr: int) -> torch.Tensor:
    """Resample audio to target sample rate."""
    if sr == target_sr:
        return waveform
    
    try:
        import torchaudio
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        return resampler(waveform)
    except:
        pass
    
    # Simple linear interpolation fallback
    ratio = target_sr / sr
    new_length = int(waveform.shape[1] * ratio)
    indices = torch.linspace(0, waveform.shape[1] - 1, new_length)
    indices_floor = indices.long().clamp(0, waveform.shape[1] - 1)
    return waveform[:, indices_floor]


@torch.no_grad()
def encode_audio_moshi(
    audio_path: str,
    mimi_model,
    device: str = "cpu",
    target_sr: int = 24000,
) -> Optional[torch.Tensor]:
    """
    Encode audio using moshi's Mimi codec.
    
    Returns tokens with shape [T, num_codebooks] where num_codebooks
    is set via mimi_model.set_num_codebooks() (default: 4).
    """
    # Load audio
    waveform, sr = load_audio(audio_path)
    
    # Resample if needed (Mimi expects 24kHz)
    waveform = resample_audio(waveform, sr, target_sr)
    
    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    
    # Add batch dimension
    waveform = waveform.unsqueeze(0).to(device)  # [1, 1, T]
    
    # Encode - returns [B, K, T] where K is set by set_num_codebooks()
    with torch.no_grad():
        codes = mimi_model.encode(waveform)
    
    # Get actual number of codebooks from output
    num_codebooks = codes.shape[1]
    
    # Reshape to [T, num_codebooks]
    codes = codes.squeeze(0).transpose(0, 1)  # [T, K]
    
    return codes.cpu()


@torch.no_grad()
def encode_audio_transformers(
    audio_path: str,
    mimi_tuple,
    device: str = "cpu",
    target_sr: int = 24000,
) -> Optional[torch.Tensor]:
    """
    Encode audio using HuggingFace Mimi.
    
    Uses reduced number of codebooks (4) for smaller TTS model.
    """
    model, feature_extractor, num_codebooks = mimi_tuple
    
    # Load audio
    waveform, sr = load_audio(audio_path)
    
    # Resample if needed
    waveform = resample_audio(waveform, sr, target_sr)
    
    # Convert to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    
    # Prepare input
    inputs = feature_extractor(
        waveform.squeeze(0).numpy(),
        sampling_rate=target_sr,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Encode with reduced quantizers (4 codebooks for small TTS)
    with torch.no_grad():
        encoder_outputs = model.encode(**inputs, num_quantizers=num_codebooks)
        codes = encoder_outputs.audio_codes  # [B, num_codebooks, T]
    
    # Reshape to [T, num_codebooks]
    codes = codes.squeeze(0).transpose(0, 1)  # [T, K]
    
    return codes.cpu()


@torch.no_grad()
def encode_audio_dummy(
    audio_path: str,
    target_sr: int = 24000,
    frame_rate: float = 12.5,
    num_codebooks: int = NUM_CODEBOOKS,
) -> torch.Tensor:
    """
    Create dummy tokens for testing when Mimi is not available.
    
    Generates random tokens based on audio duration.
    Uses 4 codebooks by default for small TTS model.
    """
    # Load audio to get duration
    waveform, sr = load_audio(audio_path)
    duration = waveform.shape[1] / sr
    
    # Calculate number of frames (12.5 Hz)
    num_frames = int(duration * frame_rate)
    num_frames = max(1, num_frames)  # At least 1 frame
    
    # Generate random tokens (3-2047, reserving 0-2 for special tokens)
    tokens = torch.randint(3, 2048, (num_frames, num_codebooks), dtype=torch.long)
    
    return tokens


def encode_audio(
    audio_path: str,
    mimi_model,
    codec_type: str,
    device: str = "cpu",
) -> Optional[torch.Tensor]:
    """
    Encode audio to Mimi tokens.
    
    Returns tokens with shape [T, num_codebooks] where num_codebooks=4.
    """
    try:
        if codec_type == "moshi":
            return encode_audio_moshi(audio_path, mimi_model, device)
        elif codec_type == "transformers":
            return encode_audio_transformers(audio_path, mimi_model, device)
        else:
            return encode_audio_dummy(audio_path)
    except Exception as e:
        print(f"Error encoding {audio_path}: {e}")
        import traceback
        traceback.print_exc()
        return None


def process_conversation(
    conv_dir: Path,
    language: str,
    output_dir: Path,
    mimi_model,
    codec_type: str,
    device: str,
) -> List[Sample]:
    """Process a single conversation directory."""
    samples = []
    conv_id = conv_dir.name
    segments_dir = conv_dir / "segments"
    
    # Find transcript
    transcript_files = list(segments_dir.glob("*.txt"))
    if not transcript_files:
        transcript_files = list(conv_dir.glob("*.txt"))
    
    if not transcript_files:
        print(f"No transcript found in {conv_dir}")
        return samples
    
    transcript_path = transcript_files[0]
    
    # Parse transcript
    lines = parse_transcript(transcript_path)
    if not lines:
        print(f"Empty transcript: {transcript_path}")
        return samples
    
    # Create tokens directory
    tokens_dir = output_dir / "tokens"
    tokens_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each line
    for idx, (speaker_num, text) in enumerate(lines, start=1):
        # Find corresponding audio
        audio_path = find_audio_for_segment(segments_dir, idx)
        if audio_path is None:
            print(f"No audio for segment {idx} in {conv_dir}")
            continue
        
        # Map speaker number to ID (1 → 0, 2 → 1)
        speaker_id = speaker_num - 1
        if speaker_id not in [0, 1]:
            print(f"Unexpected speaker number {speaker_num}, mapping to 0")
            speaker_id = 0
        
        # Encode audio
        tokens = encode_audio(str(audio_path), mimi_model, codec_type, device)
        if tokens is None:
            continue
        
        # Save tokens
        token_filename = f"{language}_{conv_id}_{idx:03d}.pt"
        token_path = tokens_dir / token_filename
        torch.save(tokens, token_path)
        
        # Create sample
        sample = Sample(
            text=text,
            speaker_id=speaker_id,
            audio_path=str(audio_path),
            tokens_path=f"tokens/{token_filename}",
            language=language,
            conversation_id=conv_id,
            segment_idx=idx,
        )
        samples.append(sample)
    
    return samples


def process_dataset(
    input_dir: str,
    output_dir: str,
    val_ratio: float = 0.1,
    device: str = "cpu",
    use_dummy: bool = False,
    num_codebooks: int = NUM_CODEBOOKS,
) -> Tuple[int, int]:
    """
    Process entire dataset.
    
    Args:
        input_dir: Path to input dataset
        output_dir: Path to output directory
        val_ratio: Fraction of data to use for validation
        device: Device for encoding (cpu/cuda/mps)
        use_dummy: If True, use dummy tokens (for testing without Mimi)
        num_codebooks: Number of codebooks to use (default: 4)
    
    Returns:
        Tuple of (num_train, num_val)
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load Mimi codec
    if use_dummy:
        mimi_model = None
        codec_type = "dummy"
        print(f"Using dummy tokens with {num_codebooks} codebooks (no Mimi codec)")
    else:
        print(f"Loading Mimi codec with {num_codebooks} codebooks...")
        mimi_model, codec_type = load_mimi_codec(device, num_codebooks)
        if mimi_model is None:
            print("Warning: Could not load Mimi codec. Using dummy tokens.")
            print("Install moshi: pip install moshi")
            print("Or transformers: pip install transformers")
            codec_type = "dummy"
        else:
            print(f"Loaded Mimi codec ({codec_type}) with {num_codebooks} codebooks")
    
    # Find all conversations
    all_samples = []
    
    # Iterate over languages
    for lang_dir in input_path.iterdir():
        if not lang_dir.is_dir():
            continue
        
        language = lang_dir.name
        print(f"\nProcessing language: {language}")
        
        # Iterate over conversations
        conv_dirs = [d for d in lang_dir.iterdir() if d.is_dir()]
        
        for conv_dir in tqdm(conv_dirs, desc=f"Conversations ({language})"):
            samples = process_conversation(
                conv_dir=conv_dir,
                language=language,
                output_dir=output_path,
                mimi_model=mimi_model,
                codec_type=codec_type,
                device=device,
            )
            all_samples.extend(samples)
    
    print(f"\nTotal samples: {len(all_samples)}")
    
    if len(all_samples) == 0:
        print("No samples processed!")
        return 0, 0
    
    # Shuffle and split
    random.shuffle(all_samples)
    split_idx = int(len(all_samples) * (1 - val_ratio))
    train_samples = all_samples[:split_idx]
    val_samples = all_samples[split_idx:]
    
    # Write manifests
    train_manifest = output_path / "train.jsonl"
    val_manifest = output_path / "val.jsonl"
    
    def write_manifest(samples: List[Sample], path: Path):
        with open(path, 'w', encoding='utf-8') as f:
            for sample in samples:
                entry = {
                    "text": sample.text,
                    "speaker_id": sample.speaker_id,
                    "audio_tokens_path": sample.tokens_path,
                    "language": sample.language,
                    "conversation_id": sample.conversation_id,
                    "segment_idx": sample.segment_idx,
                }
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    
    write_manifest(train_samples, train_manifest)
    write_manifest(val_samples, val_manifest)
    
    print(f"\nDataset prepared:")
    print(f"  Train samples: {len(train_samples)} ({train_manifest})")
    print(f"  Val samples: {len(val_samples)} ({val_manifest})")
    print(f"  Tokens dir: {output_path / 'tokens'}")
    print(f"  Codebooks: {num_codebooks}")
    
    # Print speaker distribution
    train_speakers = [s.speaker_id for s in train_samples]
    print(f"\nSpeaker distribution (train):")
    print(f"  Speaker 0: {train_speakers.count(0)}")
    print(f"  Speaker 1: {train_speakers.count(1)}")
    
    return len(train_samples), len(val_samples)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare dataset for Small Streaming TTS training"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Input dataset directory",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="data/prepared",
        help="Output directory for prepared dataset",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation split ratio (default: 0.1)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for Mimi encoding (cpu/cuda/mps)",
    )
    parser.add_argument(
        "--num-codebooks",
        type=int,
        default=NUM_CODEBOOKS,
        help=f"Number of codebooks to use (default: {NUM_CODEBOOKS})",
    )
    parser.add_argument(
        "--dummy",
        action="store_true",
        help="Use dummy tokens (for testing without Mimi)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    
    args = parser.parse_args()
    
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    print("=" * 60)
    print("Small Streaming TTS - Dataset Preparation")
    print("=" * 60)
    print(f"Input directory: {args.input}")
    print(f"Output directory: {args.output}")
    print(f"Number of codebooks: {args.num_codebooks}")
    print(f"Device: {args.device}")
    print("=" * 60)
    
    num_train, num_val = process_dataset(
        input_dir=args.input,
        output_dir=args.output,
        val_ratio=args.val_ratio,
        device=args.device,
        use_dummy=args.dummy,
        num_codebooks=args.num_codebooks,
    )
    
    if num_train > 0:
        print("\n" + "=" * 60)
        print("Done! To train, run:")
        print(f"  python -m small_tts.train --data_dir {args.output}")
        print("=" * 60)


if __name__ == "__main__":
    main()

