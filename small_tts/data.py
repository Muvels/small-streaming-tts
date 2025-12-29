"""Data loading pipeline for Small Streaming TTS.

Handles:
- Loading pre-encoded Mimi codec tokens
- Text tokenization
- Batching with variable-length sequences

Dataset format (from prepare_dataset.py):
    data_dir/
    ├── train.jsonl
    ├── val.jsonl
    └── tokens/
        ├── de_conv1_001.pt  # [T, 4] tensor
        └── ...

JSONL format:
    {"text": "...", "speaker_id": 0, "audio_tokens_path": "tokens/xxx.pt", "language": "de", ...}
"""

import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json
import logging
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)


# Language mapping
LANGUAGE_TO_ID = {
    "en": 0,
    "english": 0,
    "de": 1,
    "german": 1,
    "deutsch": 1,
}


class TextTokenizer:
    """Simple character-level tokenizer.
    
    For production, consider using a proper tokenizer like
    SentencePiece or a phoneme-based approach.
    """
    
    def __init__(self, vocab_size: int = 4096):
        self.vocab_size = vocab_size
        
        # Special tokens
        self.pad_token = 0
        self.bos_token = 1
        self.eos_token = 2
        self.unk_token = 3
        
        # Build character vocabulary
        # Basic ASCII + extended for German umlauts
        self.char_to_id = {
            "<pad>": 0,
            "<bos>": 1,
            "<eos>": 2,
            "<unk>": 3,
        }
        
        # Add printable ASCII
        for i in range(32, 127):
            self.char_to_id[chr(i)] = len(self.char_to_id)
            
        # Add German-specific characters
        for char in "äöüÄÖÜßéèêëàâçôûùîïœ–—„""'":
            if char not in self.char_to_id:
                self.char_to_id[char] = len(self.char_to_id)
            
        self.id_to_char = {v: k for k, v in self.char_to_id.items()}
        
    def encode(self, text: str, add_special: bool = True) -> List[int]:
        """Encode text to token IDs."""
        tokens = []
        if add_special:
            tokens.append(self.bos_token)
            
        for char in text:
            tokens.append(self.char_to_id.get(char, self.unk_token))
            
        if add_special:
            tokens.append(self.eos_token)
            
        return tokens
    
    def decode(self, tokens: List[int]) -> str:
        """Decode token IDs to text."""
        chars = []
        for token in tokens:
            if token in [self.pad_token, self.bos_token, self.eos_token]:
                continue
            chars.append(self.id_to_char.get(token, "?"))
        return "".join(chars)
    
    def __len__(self) -> int:
        return len(self.char_to_id)


class PreEncodedTTSDataset(Dataset):
    """Dataset for TTS training with pre-encoded Mimi tokens.
    
    Expects data in format from prepare_dataset.py:
    - data_dir/
        - train.jsonl or val.jsonl
        - tokens/
            - *.pt files with [T, num_codebooks] tensors
    
    JSONL format:
    {
        "text": "Hello world",
        "speaker_id": 0,
        "audio_tokens_path": "tokens/xxx.pt",
        "language": "de",
        "conversation_id": "...",
        "segment_idx": 1
    }
    """
    
    def __init__(
        self,
        data_dir: str,
        manifest_file: str,  # "train.jsonl" or "val.jsonl"
        tokenizer: TextTokenizer,
        max_audio_frames: int = 1000,  # ~80 seconds at 12.5 Hz
        min_audio_frames: int = 5,     # ~0.4 seconds
        num_codebooks: int = 4,
    ):
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.max_audio_frames = max_audio_frames
        self.min_audio_frames = min_audio_frames
        self.num_codebooks = num_codebooks
        
        # Load manifest
        self.samples = self._load_manifest(manifest_file)
        logger.info(f"Loaded {len(self.samples)} samples from {data_dir}/{manifest_file}")
        
    def _load_manifest(self, manifest_file: str) -> List[Dict[str, Any]]:
        """Load JSONL manifest file."""
        manifest_path = self.data_dir / manifest_file
        
        if not manifest_path.exists():
            logger.error(f"Manifest not found: {manifest_path}")
            return []
        
        samples = []
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    sample = json.loads(line)
                    samples.append(sample)
                except json.JSONDecodeError as e:
                    logger.warning(f"Error parsing line {line_num}: {e}")
                    
        return samples
    
    def _load_tokens(self, tokens_path: str) -> Optional[torch.Tensor]:
        """Load pre-encoded audio tokens."""
        full_path = self.data_dir / tokens_path
        
        if not full_path.exists():
            logger.warning(f"Token file not found: {full_path}")
            return None
            
        try:
            tokens = torch.load(full_path, weights_only=True)
            # Expected shape: [T, num_codebooks] from prepare_dataset.py
            # Need to transpose to [num_codebooks, T] for training
            if tokens.dim() == 2:
                if tokens.shape[1] == self.num_codebooks:
                    # Shape is [T, num_codebooks], transpose
                    tokens = tokens.t()  # Now [num_codebooks, T]
                # else assume it's already [num_codebooks, T]
            return tokens
        except Exception as e:
            logger.warning(f"Error loading {tokens_path}: {e}")
            return None
    
    def _get_language_id(self, language: str) -> int:
        """Convert language string to ID."""
        return LANGUAGE_TO_ID.get(language.lower(), 0)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Optional[Dict[str, torch.Tensor]]:
        sample = self.samples[idx]
        
        # Load pre-encoded tokens
        tokens = self._load_tokens(sample["audio_tokens_path"])
        if tokens is None:
            # Return a dummy sample that will be filtered
            return None
        
        # Filter by length
        num_frames = tokens.shape[1]
        if num_frames < self.min_audio_frames or num_frames > self.max_audio_frames:
            return None
        
        # Tokenize text
        text_tokens = self.tokenizer.encode(sample["text"])
        text_tokens = torch.tensor(text_tokens, dtype=torch.long)
        
        # Get speaker and language IDs
        speaker_id = sample.get("speaker_id", 0)
        language = sample.get("language", "en")
        language_id = self._get_language_id(language)
        
        return {
            "text_tokens": text_tokens,
            "audio_tokens": tokens,  # [num_codebooks, T]
            "speaker_id": torch.tensor(speaker_id, dtype=torch.long),
            "language_id": torch.tensor(language_id, dtype=torch.long),
        }


def collate_fn(batch: List[Optional[Dict[str, torch.Tensor]]]) -> Optional[Dict[str, torch.Tensor]]:
    """Collate function for variable-length sequences.
    
    Filters out None samples and pads to max length in batch.
    """
    # Filter out None samples
    batch = [item for item in batch if item is not None]
    
    if len(batch) == 0:
        return None
    
    # Find max lengths
    max_text_len = max(item["text_tokens"].shape[0] for item in batch)
    max_audio_len = max(item["audio_tokens"].shape[1] for item in batch)
    num_codebooks = batch[0]["audio_tokens"].shape[0]
    
    batch_size = len(batch)
    
    # Initialize padded tensors
    text_tokens = torch.zeros(batch_size, max_text_len, dtype=torch.long)
    audio_tokens = torch.zeros(batch_size, num_codebooks, max_audio_len, dtype=torch.long)
    speaker_ids = torch.zeros(batch_size, dtype=torch.long)
    language_ids = torch.zeros(batch_size, dtype=torch.long)
    text_lengths = torch.zeros(batch_size, dtype=torch.long)
    audio_lengths = torch.zeros(batch_size, dtype=torch.long)
    
    # Fill tensors
    for i, item in enumerate(batch):
        text_len = item["text_tokens"].shape[0]
        audio_len = item["audio_tokens"].shape[1]
        
        text_tokens[i, :text_len] = item["text_tokens"]
        audio_tokens[i, :, :audio_len] = item["audio_tokens"]
        speaker_ids[i] = item["speaker_id"]
        language_ids[i] = item["language_id"]
        text_lengths[i] = text_len
        audio_lengths[i] = audio_len
    
    return {
        "text_tokens": text_tokens,
        "audio_tokens": audio_tokens,
        "speaker_id": speaker_ids,
        "language_id": language_ids,
        "text_lengths": text_lengths,
        "audio_lengths": audio_lengths,
    }


def create_dataloader(
    data_dir: str,
    manifest_file: str,
    tokenizer: TextTokenizer,
    batch_size: int = 32,
    num_workers: int = 4,
    shuffle: bool = True,
    num_codebooks: int = 4,
    max_audio_frames: int = 1000,
    pin_memory: bool = True,
) -> DataLoader:
    """Create a DataLoader for TTS training with pre-encoded tokens.
    
    Args:
        data_dir: Directory containing manifest and tokens
        manifest_file: Name of JSONL manifest file (e.g., "train.jsonl")
        tokenizer: Text tokenizer
        batch_size: Batch size
        num_workers: Number of data loading workers
        shuffle: Whether to shuffle data
        num_codebooks: Number of audio codebooks
        max_audio_frames: Maximum audio frames per sample
        pin_memory: Whether to pin memory (set False for MPS)
    """
    dataset = PreEncodedTTSDataset(
        data_dir=data_dir,
        manifest_file=manifest_file,
        tokenizer=tokenizer,
        max_audio_frames=max_audio_frames,
        num_codebooks=num_codebooks,
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )


def create_train_val_dataloaders(
    data_dir: str,
    tokenizer: TextTokenizer,
    batch_size: int = 32,
    num_workers: int = 4,
    num_codebooks: int = 4,
    max_audio_frames: int = 1000,
    pin_memory: bool = True,
) -> tuple:
    """Create train and validation dataloaders.
    
    Args:
        data_dir: Directory containing train.jsonl, val.jsonl, and tokens/
        tokenizer: Text tokenizer
        batch_size: Batch size
        num_workers: Number of data loading workers
        num_codebooks: Number of audio codebooks
        max_audio_frames: Maximum audio frames per sample
        pin_memory: Whether to pin memory (set False for MPS)
        
    Returns:
        (train_dataloader, val_dataloader) tuple
    """
    train_loader = create_dataloader(
        data_dir=data_dir,
        manifest_file="train.jsonl",
        tokenizer=tokenizer,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        num_codebooks=num_codebooks,
        max_audio_frames=max_audio_frames,
        pin_memory=pin_memory,
    )
    
    # Check if val.jsonl exists
    val_path = Path(data_dir) / "val.jsonl"
    if val_path.exists():
        val_loader = create_dataloader(
            data_dir=data_dir,
            manifest_file="val.jsonl",
            tokenizer=tokenizer,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            num_codebooks=num_codebooks,
            max_audio_frames=max_audio_frames,
            pin_memory=pin_memory,
        )
    else:
        val_loader = None
        logger.warning(f"No val.jsonl found in {data_dir}")
    
    return train_loader, val_loader


# Keep old classes for backward compatibility
class TTSDataset(PreEncodedTTSDataset):
    """Alias for backward compatibility."""
    pass
