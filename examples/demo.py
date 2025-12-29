#!/usr/bin/env python3
"""Demo script for Small Streaming TTS.

Shows how to:
1. Create and initialize the model
2. Generate audio from text (non-streaming)
3. Generate audio in streaming fashion
4. Export for C++ deployment
"""

import torch
import soundfile as sf
from pathlib import Path
import sys

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from small_tts import TTSConfig, StreamingTTS, StreamingTTSInference


def demo_model_creation():
    """Demonstrate model creation and parameter counting."""
    print("=" * 60)
    print("Creating Small Streaming TTS Model")
    print("=" * 60)
    
    # Create config (uses defaults)
    config = TTSConfig()
    
    # Create model
    model = StreamingTTS(config)
    
    # Count parameters
    params = model.count_parameters()
    print(f"\nModel Parameters:")
    print(f"  Main Transformer: {params['main_transformer']:,}")
    print(f"  Depth Transformer: {params['depth_transformer']:,}")
    print(f"  Total: {params['total']:,}")
    print(f"  Size (MB, fp32): {params['total'] * 4 / 1024 / 1024:.1f}")
    
    return model, config


def demo_generate_audio(model: StreamingTTS):
    """Demonstrate audio generation."""
    print("\n" + "=" * 60)
    print("Generating Audio")
    print("=" * 60)
    
    device = "cpu"
    model = model.to(device).eval()
    
    # Create inference wrapper
    inference = StreamingTTSInference(model, device=device)
    
    # Test text
    text = "Hello, this is a test of the small streaming text to speech model."
    print(f"\nInput text: {text}")
    
    # Generate audio
    print("Generating audio...")
    with torch.no_grad():
        audio = inference.synthesize(
            text=text,
            speaker_id=0,  # Female
            language_id=0,  # English
            max_duration=10.0,
            temperature=0.7,
        )
    
    print(f"Generated audio shape: {audio.shape}")
    print(f"Duration: {audio.shape[0] / 24000:.2f}s")
    
    # Save audio
    output_path = Path("output/demo.wav")
    output_path.parent.mkdir(exist_ok=True)
    sf.write(str(output_path), audio.numpy(), 24000)
    print(f"Saved to: {output_path}")
    
    return audio


def demo_streaming_generation(model: StreamingTTS):
    """Demonstrate streaming audio generation."""
    print("\n" + "=" * 60)
    print("Streaming Generation Demo")
    print("=" * 60)
    
    device = "cpu"
    model = model.to(device).eval()
    
    inference = StreamingTTSInference(model, device=device)
    
    text = "This demonstrates streaming text to speech generation."
    print(f"\nInput text: {text}")
    print("\nGenerating audio chunks...")
    
    chunks = []
    frame_count = 0
    max_frames = 50  # Limit for demo
    
    with torch.no_grad():
        for chunk in inference.synthesize_streaming(
            text=text,
            speaker_id=1,  # Male
            language_id=1,  # German
            temperature=0.7,
        ):
            chunks.append(chunk)
            frame_count += 1
            
            if frame_count % 10 == 0:
                print(f"  Generated {frame_count} frames ({frame_count * 80}ms)")
                
            if frame_count >= max_frames:
                print(f"  Stopped at {max_frames} frames for demo")
                break
    
    # Concatenate chunks
    audio = torch.cat(chunks, dim=0)
    print(f"\nTotal frames: {frame_count}")
    print(f"Total audio: {audio.shape[0]} samples ({audio.shape[0] / 24000:.2f}s)")
    
    return audio


def demo_export(model: StreamingTTS, config: TTSConfig):
    """Demonstrate model export for C++ deployment."""
    print("\n" + "=" * 60)
    print("Export Demo")
    print("=" * 60)
    
    from small_tts.export import BinaryExporter, generate_cpp_header
    
    # Create fake checkpoint for demo
    output_dir = Path("output/export")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Export to binary format
    print("\nExporting to binary format...")
    exporter = BinaryExporter(quantize=False)
    exporter.export(
        model=model,
        config=config.to_dict(),
        output_path=str(output_dir / "model.bin"),
    )
    
    # Export quantized version
    print("Exporting quantized (fp16) version...")
    exporter_q = BinaryExporter(quantize=True)
    exporter_q.export(
        model=model,
        config=config.to_dict(),
        output_path=str(output_dir / "model_fp16.bin"),
    )
    
    # Generate C++ header
    print("Generating C++ header...")
    generate_cpp_header(config.to_dict(), str(output_dir / "small_tts.h"))
    
    # Print file sizes
    print("\nExported files:")
    for f in output_dir.iterdir():
        size_mb = f.stat().st_size / 1024 / 1024
        print(f"  {f.name}: {size_mb:.1f} MB")


def main():
    print("Small Streaming TTS Demo")
    print("=" * 60)
    
    # Create model
    model, config = demo_model_creation()
    
    # Generate audio (non-streaming)
    demo_generate_audio(model)
    
    # Streaming generation
    demo_streaming_generation(model)
    
    # Export for C++
    demo_export(model, config)
    
    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()


