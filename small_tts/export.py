"""Export utilities for Small Streaming TTS.

Provides weight export in formats suitable for C++ inference:
1. Simple binary format (custom, minimal dependencies)
2. ONNX format (industry standard)

The binary format is designed for easy parsing in C++:
- Header with model config
- Sequential weight tensors with shape metadata
"""

import torch
import torch.nn as nn
import struct
import json
from pathlib import Path
from typing import Dict, Any, Optional, BinaryIO
import logging

logger = logging.getLogger(__name__)


# Binary format constants
MAGIC_NUMBER = b"STTS"  # Small TTS
FORMAT_VERSION = 1


class BinaryExporter:
    """Export model weights to simple binary format.
    
    Format:
    - Magic number (4 bytes): "STTS"
    - Version (4 bytes, uint32)
    - Config JSON length (4 bytes, uint32)
    - Config JSON (variable)
    - Number of tensors (4 bytes, uint32)
    - For each tensor:
        - Name length (4 bytes, uint32)
        - Name (variable, utf-8)
        - Number of dimensions (4 bytes, uint32)
        - Shape (dims * 4 bytes, uint32 each)
        - Data type (4 bytes): 0=float32, 1=float16, 2=int32, 3=int64
        - Data (variable, row-major order)
    """
    
    DTYPE_MAP = {
        torch.float32: 0,
        torch.float16: 1,
        torch.int32: 2,
        torch.int64: 3,
    }
    
    DTYPE_SIZE = {
        0: 4,  # float32
        1: 2,  # float16
        2: 4,  # int32
        3: 8,  # int64
    }
    
    def __init__(self, quantize: bool = False):
        self.quantize = quantize
        
    def export(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        output_path: str,
    ):
        """Export model to binary format.
        
        Args:
            model: PyTorch model
            config: Model configuration dict
            output_path: Output file path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "wb") as f:
            self._write_header(f, config)
            self._write_tensors(f, model)
            
        logger.info(f"Exported model to {output_path}")
        
        # Also export tensor list for reference
        self._export_tensor_list(model, output_path.with_suffix(".json"))
        
    def _write_header(self, f: BinaryIO, config: Dict[str, Any]):
        """Write file header."""
        # Magic number
        f.write(MAGIC_NUMBER)
        
        # Version
        f.write(struct.pack("<I", FORMAT_VERSION))
        
        # Config JSON
        config_json = json.dumps(config, indent=2).encode("utf-8")
        f.write(struct.pack("<I", len(config_json)))
        f.write(config_json)
        
    def _write_tensors(self, f: BinaryIO, model: nn.Module):
        """Write all model tensors."""
        state_dict = model.state_dict()
        
        # Write number of tensors
        f.write(struct.pack("<I", len(state_dict)))
        
        for name, tensor in state_dict.items():
            self._write_tensor(f, name, tensor)
            
    def _write_tensor(self, f: BinaryIO, name: str, tensor: torch.Tensor):
        """Write single tensor."""
        # Ensure tensor is contiguous and on CPU
        tensor = tensor.contiguous().cpu()
        
        # Optionally quantize to float16
        if self.quantize and tensor.dtype == torch.float32:
            tensor = tensor.half()
            
        # Name
        name_bytes = name.encode("utf-8")
        f.write(struct.pack("<I", len(name_bytes)))
        f.write(name_bytes)
        
        # Shape
        shape = tensor.shape
        f.write(struct.pack("<I", len(shape)))
        for dim in shape:
            f.write(struct.pack("<I", dim))
            
        # Data type
        dtype_id = self.DTYPE_MAP.get(tensor.dtype, 0)
        f.write(struct.pack("<I", dtype_id))
        
        # Data
        f.write(tensor.numpy().tobytes())
        
    def _export_tensor_list(self, model: nn.Module, output_path: Path):
        """Export list of tensors with metadata for debugging."""
        state_dict = model.state_dict()
        
        tensor_info = []
        for name, tensor in state_dict.items():
            tensor_info.append({
                "name": name,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "numel": tensor.numel(),
                "size_bytes": tensor.numel() * tensor.element_size(),
            })
            
        with open(output_path, "w") as f:
            json.dump(tensor_info, f, indent=2)
            
        logger.info(f"Exported tensor list to {output_path}")


class ONNXExporter:
    """Export model to ONNX format.
    
    ONNX is a standard format with C++ runtime support via ONNX Runtime.
    This exporter handles:
    - Static graph export
    - Dynamic axes for variable sequence lengths
    - Quantization options
    """
    
    def __init__(
        self,
        opset_version: int = 17,
        dynamic_axes: bool = True,
    ):
        self.opset_version = opset_version
        self.dynamic_axes = dynamic_axes
        
    def export_main_transformer(
        self,
        model: nn.Module,
        output_path: str,
        batch_size: int = 1,
        text_seq_len: int = 128,
        audio_seq_len: int = 256,
    ):
        """Export main transformer to ONNX.
        
        Note: Due to KV cache complexity, we export without caching.
        For production streaming, use the binary format with custom C++ inference.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        model = model.cpu().eval()
        
        # Create dummy inputs
        text_tokens = torch.randint(0, 4096, (batch_size, text_seq_len))
        audio_tokens = torch.randint(0, 2048, (batch_size, audio_seq_len))
        speaker_id = torch.zeros(batch_size, dtype=torch.long)
        language_id = torch.zeros(batch_size, dtype=torch.long)
        
        # Dynamic axes
        dynamic_axes = None
        if self.dynamic_axes:
            dynamic_axes = {
                "text_tokens": {0: "batch", 1: "text_seq"},
                "audio_tokens": {0: "batch", 1: "audio_seq"},
                "logits": {0: "batch", 1: "seq"},
                "hidden": {0: "batch", 1: "seq"},
            }
        
        # Export
        torch.onnx.export(
            model,
            (text_tokens, audio_tokens, speaker_id, language_id),
            str(output_path),
            input_names=["text_tokens", "audio_tokens", "speaker_id", "language_id"],
            output_names=["logits", "hidden"],
            dynamic_axes=dynamic_axes,
            opset_version=self.opset_version,
            do_constant_folding=True,
        )
        
        logger.info(f"Exported ONNX model to {output_path}")
        
    def export_depth_transformer(
        self,
        model: nn.Module,
        output_path: str,
        batch_size: int = 1,
    ):
        """Export depth transformer to ONNX."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        model = model.cpu().eval()
        
        # Dummy inputs
        cb1_tokens = torch.randint(0, 2048, (batch_size, 1))
        main_hidden = torch.randn(batch_size, 1, 768)
        
        dynamic_axes = None
        if self.dynamic_axes:
            dynamic_axes = {
                "cb1_tokens": {0: "batch"},
                "main_hidden": {0: "batch"},
                "logits": {0: "batch"},
            }
        
        # Wrap forward to return only logits
        class DepthWrapper(nn.Module):
            def __init__(self, depth_model):
                super().__init__()
                self.model = depth_model
                
            def forward(self, cb1_tokens, main_hidden):
                logits, _ = self.model(cb1_tokens, main_hidden)
                return logits
        
        wrapper = DepthWrapper(model)
        
        torch.onnx.export(
            wrapper,
            (cb1_tokens, main_hidden),
            str(output_path),
            input_names=["cb1_tokens", "main_hidden"],
            output_names=["logits"],
            dynamic_axes=dynamic_axes,
            opset_version=self.opset_version,
        )
        
        logger.info(f"Exported depth ONNX model to {output_path}")


def export_for_cpp(
    checkpoint_path: str,
    output_dir: str,
    format: str = "binary",  # "binary" or "onnx"
    quantize: bool = False,
):
    """Export trained model for C++ inference.
    
    Args:
        checkpoint_path: Path to training checkpoint
        output_dir: Output directory
        format: Export format ("binary" or "onnx")
        quantize: Whether to quantize to float16
    """
    from small_tts.config import TTSConfig
    from small_tts.model import StreamingTTS
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load checkpoint
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # Recreate model from config
    config_dict = checkpoint["config"]
    config = TTSConfig()
    # Restore config values
    for key, value in config_dict.get("main", {}).items():
        setattr(config.main, key, value)
    for key, value in config_dict.get("depth", {}).items():
        setattr(config.depth, key, value)
    for key, value in config_dict.items():
        if not isinstance(value, dict):
            if hasattr(config, key):
                setattr(config, key, value)
    
    model = StreamingTTS(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    if format == "binary":
        exporter = BinaryExporter(quantize=quantize)
        exporter.export(
            model=model,
            config=config.to_dict(),
            output_path=str(output_dir / "model.bin"),
        )
    elif format == "onnx":
        exporter = ONNXExporter()
        exporter.export_main_transformer(
            model=model.main_transformer,
            output_path=str(output_dir / "main_transformer.onnx"),
        )
        exporter.export_depth_transformer(
            model=model.depth_transformer,
            output_path=str(output_dir / "depth_transformer.onnx"),
        )
    else:
        raise ValueError(f"Unknown format: {format}")
    
    # Also save config as JSON
    with open(output_dir / "config.json", "w") as f:
        json.dump(config.to_dict(), f, indent=2)
        
    logger.info(f"Export complete. Files saved to {output_dir}")


# C++ header generation for binary format parsing
CPP_HEADER_TEMPLATE = '''
#pragma once

// Auto-generated header for Small Streaming TTS model loading
// Format version: {version}

#include <cstdint>
#include <string>
#include <vector>
#include <fstream>
#include <map>

namespace small_tts {{

// Data types
enum class DType : uint32_t {{
    Float32 = 0,
    Float16 = 1,
    Int32 = 2,
    Int64 = 3,
}};

// Tensor metadata
struct TensorMeta {{
    std::string name;
    std::vector<uint32_t> shape;
    DType dtype;
    size_t offset;  // Offset in data buffer
    size_t size_bytes;
}};

// Model config
struct ModelConfig {{
    // Main transformer
    int hidden_dim = {hidden_dim};
    int num_layers = {num_layers};
    int num_heads = {num_heads};
    int num_kv_heads = {num_kv_heads};
    int ffn_dim = {ffn_dim};
    
    // Depth transformer
    int depth_hidden_dim = {depth_hidden_dim};
    int depth_num_layers = {depth_num_layers};
    
    // Vocabulary
    int text_vocab_size = {text_vocab_size};
    int audio_vocab_size = {audio_vocab_size};
    int num_codebooks = {num_codebooks};
    
    // Codec
    int sample_rate = {sample_rate};
    float frame_rate = {frame_rate};
}};

class ModelLoader {{
public:
    static constexpr char MAGIC[4] = {{'S', 'T', 'T', 'S'}};
    static constexpr uint32_t VERSION = {version};
    
    bool load(const std::string& path);
    
    const ModelConfig& config() const {{ return config_; }}
    const std::vector<TensorMeta>& tensors() const {{ return tensors_; }}
    const std::vector<char>& data() const {{ return data_; }}
    
    // Get tensor data pointer
    template<typename T>
    const T* get_tensor(const std::string& name) const {{
        auto it = tensor_map_.find(name);
        if (it == tensor_map_.end()) return nullptr;
        return reinterpret_cast<const T*>(data_.data() + it->second.offset);
    }}
    
private:
    ModelConfig config_;
    std::vector<TensorMeta> tensors_;
    std::map<std::string, TensorMeta> tensor_map_;
    std::vector<char> data_;
}};

}} // namespace small_tts
'''


def generate_cpp_header(config: Dict[str, Any], output_path: str):
    """Generate C++ header file for model loading."""
    header = CPP_HEADER_TEMPLATE.format(
        version=FORMAT_VERSION,
        hidden_dim=config["main"]["hidden_dim"],
        num_layers=config["main"]["num_layers"],
        num_heads=config["main"]["num_heads"],
        num_kv_heads=config["main"]["num_kv_heads"],
        ffn_dim=config["main"]["ffn_dim"],
        depth_hidden_dim=config["depth"]["hidden_dim"],
        depth_num_layers=config["depth"]["num_layers"],
        text_vocab_size=config["text_vocab_size"],
        audio_vocab_size=config["audio_vocab_size"],
        num_codebooks=config["num_codebooks"],
        sample_rate=config["codec"]["sample_rate"],
        frame_rate=config["codec"]["frame_rate"],
    )
    
    with open(output_path, "w") as f:
        f.write(header)
        
    logger.info(f"Generated C++ header: {output_path}")


def main():
    """Command-line export tool."""
    import argparse
    
    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser(description="Export Small TTS for C++")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--format", type=str, default="binary", choices=["binary", "onnx"])
    parser.add_argument("--quantize", action="store_true")
    parser.add_argument("--generate_header", action="store_true")
    args = parser.parse_args()
    
    export_for_cpp(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        format=args.format,
        quantize=args.quantize,
    )
    
    if args.generate_header:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        generate_cpp_header(
            config=checkpoint["config"],
            output_path=str(Path(args.output_dir) / "small_tts.h"),
        )


if __name__ == "__main__":
    main()


