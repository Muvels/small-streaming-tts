#!/usr/bin/env python3
"""Inference script for trained Small Streaming TTS model.

Usage:
    python inference.py --checkpoint checkpoints/final.pt --text "Hello world"
    python inference.py --checkpoint checkpoints/final.pt --text "Hallo Welt" --language 1
"""

import torch
import soundfile as sf
from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).parent))

from small_tts.config import TTSConfig
from small_tts.model import StreamingTTS, StreamingTTSInference


def load_model_from_checkpoint(checkpoint_path: str, device: str = "cpu"):
    """Load a trained model from checkpoint."""
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Recreate config from checkpoint
    config_dict = checkpoint.get("config", {})
    config = TTSConfig()
    
    # Restore config values
    for key, value in config_dict.get("main", {}).items():
        if hasattr(config.main, key):
            setattr(config.main, key, value)
    for key, value in config_dict.get("depth", {}).items():
        if hasattr(config.depth, key):
            setattr(config.depth, key, value)
    for key, value in config_dict.items():
        if not isinstance(value, dict):
            if hasattr(config, key):
                setattr(config, key, value)
    
    # Create and load model
    model = StreamingTTS(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    # Print checkpoint info
    stage = checkpoint.get("stage", "unknown")
    epoch = checkpoint.get("epoch", "unknown")
    step = checkpoint.get("global_step", "unknown")
    print(f"  Stage: {stage}, Epoch: {epoch}, Step: {step}")
    
    # Count parameters
    params = model.count_parameters()
    print(f"  Parameters: {params['total']:,}")
    
    return model, config


def main():
    parser = argparse.ArgumentParser(description="Test TTS model inference")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/final.pt",
                        help="Path to checkpoint file")
    parser.add_argument("--text", type=str, default="Hello, this is a test.",
                        help="Text to synthesize")
    parser.add_argument("--speaker", type=int, default=0,
                        help="Speaker ID (0=female, 1=male)")
    parser.add_argument("--language", type=int, default=0,
                        help="Language ID (0=English, 1=German)")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature (lower=more deterministic)")
    parser.add_argument("--max_duration", type=float, default=10.0,
                        help="Maximum audio duration in seconds")
    parser.add_argument("--output", type=str, default="output.wav",
                        help="Output audio file path")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "mps", "cpu"],
                        help="Device to use")
    parser.add_argument("--streaming", action="store_true",
                        help="Use streaming generation (shows progress)")
    args = parser.parse_args()
    
    # Select device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device
    print(f"Using device: {device}")
    
    # Load model
    model, config = load_model_from_checkpoint(args.checkpoint, device)
    model = model.to(device)
    
    # Create inference wrapper
    inference = StreamingTTSInference(model, device=device)
    
    print(f"\nGenerating audio for: \"{args.text}\"")
    print(f"  Speaker: {args.speaker}, Language: {'English' if args.language == 0 else 'German'}")
    print(f"  Temperature: {args.temperature}")
    
    with torch.no_grad():
        if args.streaming:
            # Streaming generation with progress
            print("\nGenerating (streaming)...")
            chunks = []
            frame_count = 0
            max_frames = int(args.max_duration * 12.5)  # 12.5 fps at 80ms per frame
            
            for chunk in inference.synthesize_streaming(
                text=args.text,
                speaker_id=args.speaker,
                language_id=args.language,
                temperature=args.temperature,
            ):
                chunks.append(chunk)
                frame_count += 1
                
                if frame_count % 10 == 0:
                    duration = frame_count * 0.08  # 80ms per frame
                    print(f"  Frame {frame_count} ({duration:.1f}s)")
                
                if frame_count >= max_frames:
                    print(f"  Reached max duration ({args.max_duration}s)")
                    break
            
            audio = torch.cat(chunks, dim=0)
        else:
            # Non-streaming generation
            print("\nGenerating...")
            audio = inference.synthesize(
                text=args.text,
                speaker_id=args.speaker,
                language_id=args.language,
                max_duration=args.max_duration,
                temperature=args.temperature,
            )
    
    # Save audio
    sample_rate = config.codec.sample_rate if hasattr(config.codec, 'sample_rate') else 24000
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    audio_np = audio.cpu().numpy()
    sf.write(str(output_path), audio_np, sample_rate)
    
    duration = len(audio_np) / sample_rate
    print(f"\nSaved audio to: {output_path}")
    print(f"  Duration: {duration:.2f}s")
    print(f"  Samples: {len(audio_np):,}")
    print(f"  Sample rate: {sample_rate} Hz")


if __name__ == "__main__":
    main()

