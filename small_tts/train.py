"""Training script for Small Streaming TTS.

Implements two-stage training:
1. Stage 1: Train main transformer for CB1 prediction
2. Stage 2: Joint training with depth transformer for CB2-4

Supports:
- CUDA (NVIDIA GPUs)
- MPS (Apple Silicon)
- CPU

Usage:
    python -m small_tts.train --data_dir data/prepared --device auto
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.tensorboard import SummaryWriter

import logging
import os
from pathlib import Path
from tqdm import tqdm
from typing import Optional, Dict, Any
import time

from small_tts.config import TTSConfig
from small_tts.model import StreamingTTS
from small_tts.data import TextTokenizer, create_train_val_dataloaders

logger = logging.getLogger(__name__)


def get_device(device_str: str = "auto") -> torch.device:
    """Get the best available device.
    
    Args:
        device_str: "auto", "cuda", "mps", or "cpu"
        
    Returns:
        torch.device
    """
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    else:
        return torch.device(device_str)


def supports_amp(device: torch.device) -> bool:
    """Check if device supports automatic mixed precision."""
    if device.type == "cuda":
        return True
    elif device.type == "mps":
        # MPS has limited AMP support, disable for stability
        return False
    else:
        return False


class Trainer:
    """Two-stage trainer for Streaming TTS.
    
    Supports CUDA, MPS (Apple Silicon), and CPU.
    """
    
    def __init__(
        self,
        config: TTSConfig,
        model: StreamingTTS,
        train_dataloader,
        val_dataloader=None,
        device: torch.device = None,
        freeze_main_transformer: bool = False,
    ):
        self.config = config
        self.device = device or get_device()
        self.model = model.to(self.device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.freeze_main_transformer = freeze_main_transformer
        
        # Training state
        self.stage = 1
        self.global_step = 0
        self.epoch = 0
        
        # Mixed precision - only for CUDA
        self.use_amp = supports_amp(self.device) and config.training.mixed_precision
        if self.use_amp:
            from torch.amp import GradScaler
            self.scaler = GradScaler('cuda')
        else:
            self.scaler = None
            
        if self.device.type == "mps":
            logger.info("MPS device detected - using float32 (no AMP)")
        
        # Logging
        self.log_dir = Path(config.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(self.log_dir)
        
        # Checkpoints
        self.checkpoint_dir = self.log_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        logger.info(f"Trainer initialized on device: {self.device}")
        logger.info(f"Mixed precision: {self.use_amp}")
        if freeze_main_transformer:
            logger.info("Main transformer will be FROZEN in Stage 2")
        
    def create_optimizer(self, stage: int) -> AdamW:
        """Create optimizer for current stage."""
        if stage == 1:
            lr = self.config.training.stage1_lr
            # Only train main transformer
            params = self.model.main_transformer.parameters()
        else:
            lr = self.config.training.stage2_lr
            
            if self.freeze_main_transformer:
                # Freeze main transformer - only train depth transformer
                logger.info("Freezing main transformer parameters")
                for param in self.model.main_transformer.parameters():
                    param.requires_grad = False
                params = self.model.depth_transformer.parameters()
            else:
                # Train both transformers
                params = self.model.parameters()
            
        return AdamW(
            params,
            lr=lr,
            weight_decay=self.config.training.weight_decay,
            betas=(0.9, 0.999),
        )
    
    def create_scheduler(self, optimizer: AdamW, stage: int, num_training_steps: int):
        """Create learning rate scheduler with warmup."""
        if stage == 1:
            warmup_steps = self.config.training.stage1_warmup_steps
        else:
            warmup_steps = self.config.training.stage2_warmup_steps
        
        # Ensure warmup doesn't exceed training steps
        warmup_steps = min(warmup_steps, num_training_steps // 2)
            
        # Linear warmup
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        
        # Cosine annealing after warmup
        cosine_steps = max(1, num_training_steps - warmup_steps)
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=cosine_steps,
            eta_min=1e-6,
        )
        
        return SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps],
        )
    
    def train_step_stage1(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> tuple:
        """Training step for stage 1 (main transformer only).
        
        Only trains CB1 prediction - the semantic codebook.
        """
        text_tokens = batch["text_tokens"].to(self.device)
        audio_tokens = batch["audio_tokens"].to(self.device)
        speaker_id = batch["speaker_id"].to(self.device)
        language_id = batch["language_id"].to(self.device)
        
        # Extract CB1 tokens [batch, seq]
        cb1_tokens = audio_tokens[:, 0, :]
        
        # Teacher forcing: input is shifted right, target is original
        cb1_input = cb1_tokens[:, :-1]
        cb1_target = cb1_tokens[:, 1:]
        
        # Forward through main transformer
        if self.use_amp:
            from torch.amp import autocast
            with autocast('cuda'):
                logits, hidden, _ = self.model.main_transformer(
                    text_tokens=text_tokens,
                    audio_tokens=cb1_input,
                    speaker_id=speaker_id,
                    language_id=language_id,
                )
                
                # Get logits for audio positions only
                audio_start = 1 + text_tokens.shape[1]  # Skip cond + text
                audio_logits = logits[:, audio_start:audio_start + cb1_target.shape[1], :]
                
                # Compute loss
                loss = F.cross_entropy(
                    audio_logits.reshape(-1, self.config.audio_vocab_size),
                    cb1_target.reshape(-1),
                    reduction="mean",
                )
        else:
            logits, hidden, _ = self.model.main_transformer(
                text_tokens=text_tokens,
                audio_tokens=cb1_input,
                speaker_id=speaker_id,
                language_id=language_id,
            )
            
            audio_start = 1 + text_tokens.shape[1]
            audio_logits = logits[:, audio_start:audio_start + cb1_target.shape[1], :]
            
            loss = F.cross_entropy(
                audio_logits.reshape(-1, self.config.audio_vocab_size),
                cb1_target.reshape(-1),
                reduction="mean",
            )
        
        return {"loss": loss.item(), "cb1_loss": loss.item()}, loss
    
    def train_step_stage2(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> tuple:
        """Training step for stage 2 (joint training).
        
        Trains both main transformer (CB1) and depth transformer (CB2-4).
        """
        text_tokens = batch["text_tokens"].to(self.device)
        audio_tokens = batch["audio_tokens"].to(self.device)
        speaker_id = batch["speaker_id"].to(self.device)
        language_id = batch["language_id"].to(self.device)
        
        if self.use_amp:
            from torch.amp import autocast
            with autocast('cuda'):
                cb1_loss, depth_loss, total_loss = self.model(
                    text_tokens=text_tokens,
                    audio_tokens=audio_tokens,
                    speaker_id=speaker_id,
                    language_id=language_id,
                )
        else:
            cb1_loss, depth_loss, total_loss = self.model(
                text_tokens=text_tokens,
                audio_tokens=audio_tokens,
                speaker_id=speaker_id,
                language_id=language_id,
            )
        
        return {
            "loss": total_loss.item(),
            "cb1_loss": cb1_loss.item(),
            "depth_loss": depth_loss.item(),
        }, total_loss
    
    def train_epoch(
        self,
        optimizer: AdamW,
        scheduler,
        stage: int,
    ) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_metrics = {}
        num_batches = 0
        
        pbar = tqdm(self.train_dataloader, desc=f"Epoch {self.epoch + 1}")
        
        accumulation_steps = getattr(self.config.training, 'gradient_accumulation', 1)
        accumulated_loss = 0.0
        
        for batch_idx, batch in enumerate(pbar):
            if batch is None:
                continue
            
            # Forward pass
            if stage == 1:
                metrics, loss = self.train_step_stage1(batch)
            else:
                metrics, loss = self.train_step_stage2(batch)
            
            # Scale loss for gradient accumulation
            loss = loss / accumulation_steps
            accumulated_loss += loss.item()
            
            # Backward pass
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Step optimizer every accumulation_steps
            if (batch_idx + 1) % accumulation_steps == 0:
                if self.scaler is not None:
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.training.gradient_clip,
                    )
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.training.gradient_clip,
                    )
                    optimizer.step()
                
                optimizer.zero_grad(set_to_none=True)
                accumulated_loss = 0.0
                
            scheduler.step()
            
            # Accumulate metrics
            for key, value in metrics.items():
                total_metrics[key] = total_metrics.get(key, 0) + value
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({k: f"{v:.4f}" for k, v in metrics.items()})
            
            # Logging
            if self.global_step % 100 == 0:
                for key, value in metrics.items():
                    self.writer.add_scalar(f"train/{key}", value, self.global_step)
                self.writer.add_scalar(
                    "train/lr",
                    optimizer.param_groups[0]["lr"],
                    self.global_step,
                )
            
            # MPS memory cleanup (critical for Apple Silicon)
            if self.device.type == "mps" and self.global_step % 10 == 0:
                torch.mps.empty_cache()
                
            self.global_step += 1
        
        # End of epoch cleanup for MPS
        if self.device.type == "mps":
            torch.mps.empty_cache()
            import gc
            gc.collect()
        
        # Average metrics
        if num_batches > 0:
            avg_metrics = {k: v / num_batches for k, v in total_metrics.items()}
        else:
            avg_metrics = {"loss": 0.0}
        return avg_metrics
    
    @torch.no_grad()
    def validate(self, stage: int) -> Dict[str, float]:
        """Run validation."""
        if self.val_dataloader is None:
            return {}
            
        self.model.eval()
        
        total_metrics = {}
        num_batches = 0
        
        for batch in tqdm(self.val_dataloader, desc="Validating"):
            if batch is None:
                continue
                
            if stage == 1:
                metrics, _ = self.train_step_stage1(batch)
            else:
                metrics, _ = self.train_step_stage2(batch)
                
            for key, value in metrics.items():
                total_metrics[key] = total_metrics.get(key, 0) + value
            num_batches += 1
        
        if num_batches > 0:
            avg_metrics = {k: v / num_batches for k, v in total_metrics.items()}
        else:
            avg_metrics = {"loss": 0.0}
        
        # Log validation metrics
        for key, value in avg_metrics.items():
            self.writer.add_scalar(f"val/{key}", value, self.global_step)
            
        return avg_metrics
    
    def save_checkpoint(self, name: str):
        """Save training checkpoint."""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "config": self.config.to_dict(),
            "stage": self.stage,
            "epoch": self.epoch,
            "global_step": self.global_step,
        }
        
        path = self.checkpoint_dir / f"{name}.pt"
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, path: str, weights_only: bool = False):
        """Load training checkpoint.
        
        Args:
            path: Path to checkpoint file
            weights_only: If True, only load model weights (ignore training state).
                         Useful for starting Stage 2 from a Stage 1 checkpoint.
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        
        if not weights_only:
            self.stage = checkpoint.get("stage", 1)
            self.epoch = checkpoint.get("epoch", 0)
            self.global_step = checkpoint.get("global_step", 0)
            logger.info(f"Loaded checkpoint from {path} (full state)")
        else:
            logger.info(f"Loaded checkpoint from {path} (weights only)")
            
        # Log checkpoint info
        ckpt_stage = checkpoint.get("stage", "unknown")
        ckpt_epoch = checkpoint.get("epoch", "unknown")
        logger.info(f"  Checkpoint was from: Stage {ckpt_stage}, Epoch {ckpt_epoch}")
    
    def train_stage1(self):
        """Stage 1: Train main transformer for CB1 prediction."""
        logger.info("=" * 50)
        logger.info("Starting Stage 1: Main Transformer Training")
        logger.info("=" * 50)
        
        self.stage = 1
        num_epochs = self.config.training.stage1_epochs
        num_training_steps = len(self.train_dataloader) * num_epochs
        
        logger.info(f"Epochs: {num_epochs}, Steps: {num_training_steps}")
        
        optimizer = self.create_optimizer(stage=1)
        scheduler = self.create_scheduler(optimizer, stage=1, num_training_steps=num_training_steps)
        
        best_val_loss = float("inf")
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            
            # Train
            train_metrics = self.train_epoch(optimizer, scheduler, stage=1)
            logger.info(f"Epoch {epoch + 1}/{num_epochs} - Train: {train_metrics}")
            
            # Validate
            val_metrics = self.validate(stage=1)
            if val_metrics:
                logger.info(f"Epoch {epoch + 1}/{num_epochs} - Val: {val_metrics}")
                
                # Save best model
                if val_metrics.get("loss", float("inf")) < best_val_loss:
                    best_val_loss = val_metrics["loss"]
                    self.save_checkpoint("best_stage1")
                    
            # Save periodic checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f"stage1_epoch{epoch + 1}")
                
        self.save_checkpoint("stage1_final")
        logger.info("Stage 1 training complete!")
    
    def train_stage2(self):
        """Stage 2: Joint training with depth transformer."""
        logger.info("=" * 50)
        logger.info("Starting Stage 2: Joint Training")
        logger.info("=" * 50)
        
        self.stage = 2
        num_epochs = self.config.training.stage2_epochs
        num_training_steps = len(self.train_dataloader) * num_epochs
        
        logger.info(f"Epochs: {num_epochs}, Steps: {num_training_steps}")
        
        optimizer = self.create_optimizer(stage=2)
        scheduler = self.create_scheduler(optimizer, stage=2, num_training_steps=num_training_steps)
        
        best_val_loss = float("inf")
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            
            # Train
            train_metrics = self.train_epoch(optimizer, scheduler, stage=2)
            logger.info(f"Epoch {epoch + 1}/{num_epochs} - Train: {train_metrics}")
            
            # Validate
            val_metrics = self.validate(stage=2)
            if val_metrics:
                logger.info(f"Epoch {epoch + 1}/{num_epochs} - Val: {val_metrics}")
                
                if val_metrics.get("loss", float("inf")) < best_val_loss:
                    best_val_loss = val_metrics["loss"]
                    self.save_checkpoint("best_stage2")
                    
            # Save periodic checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f"stage2_epoch{epoch + 1}")
                
        self.save_checkpoint("final")
        logger.info("Stage 2 training complete!")
    
    def train(self, start_stage: int = 1):
        """Run two-stage training.
        
        Args:
            start_stage: Which stage to start from (1 or 2).
                        Use start_stage=2 to skip Stage 1 (e.g., when loading
                        a Stage 1 checkpoint and only training Stage 2).
        """
        start_time = time.time()
        
        if start_stage == 1:
            # Stage 1
            self.train_stage1()
            # Stage 2
            self.train_stage2()
        elif start_stage == 2:
            logger.info("Skipping Stage 1, starting directly at Stage 2")
            # Stage 2 only
            self.train_stage2()
        else:
            raise ValueError(f"start_stage must be 1 or 2, got {start_stage}")
        
        elapsed = time.time() - start_time
        logger.info(f"Training complete in {elapsed / 3600:.2f} hours")
        
        self.writer.close()


def main():
    """Main training entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Small Streaming TTS")
    parser.add_argument("--config", type=str, default="config/default.yaml",
                       help="Path to config YAML file")
    parser.add_argument("--data_dir", type=str, required=True,
                       help="Path to prepared dataset (with train.jsonl, val.jsonl, tokens/)")
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Path to checkpoint file to load model weights from")
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to checkpoint to resume training from (loads full state)")
    parser.add_argument("--start_stage", type=int, default=1, choices=[1, 2],
                       help="Which stage to start training from (default: 1)")
    parser.add_argument("--freeze_main_transformer", action="store_true",
                       help="Freeze main transformer in Stage 2 (only train depth transformer)")
    parser.add_argument("--device", type=str, default="auto",
                       choices=["auto", "cuda", "mps", "cpu"],
                       help="Device to train on")
    parser.add_argument("--batch_size", type=int, default=None,
                       help="Override batch size (default: 32 for CUDA, 8 for MPS, 4 for CPU)")
    parser.add_argument("--num_workers", type=int, default=None,
                       help="Number of data loading workers (default: 4 for CUDA, 0 for MPS)")
    parser.add_argument("--log_dir", type=str, default=None,
                       help="Override log directory from config")
    parser.add_argument("--gradient_accumulation", type=int, default=1,
                       help="Gradient accumulation steps (use to simulate larger batches)")
    parser.add_argument("--stage1_epochs", type=int, default=None,
                       help="Override number of epochs for stage 1 (main transformer)")
    parser.add_argument("--stage2_epochs", type=int, default=None,
                       help="Override number of epochs for stage 2 (joint training)")
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    # Load config
    config = TTSConfig.from_yaml(args.config)
    
    # Get device first to set device-specific defaults
    device = get_device(args.device)
    logger.info(f"Using device: {device}")
    
    # Device-specific defaults
    if device.type == "mps":
        # MPS has memory limitations - use smaller batches
        default_batch_size = 8
        default_num_workers = 0  # Multiprocessing can cause issues on MPS
        logger.info("Apple Silicon detected - using MPS-optimized settings:")
        logger.info("  - Reduced batch size (8) to manage unified memory")
        logger.info("  - Disabled data workers (0) for stability")
        logger.info("  - Disabled mixed precision (not well supported)")
    elif device.type == "cuda":
        default_batch_size = 32
        default_num_workers = 4
        logger.info(f"CUDA device: {torch.cuda.get_device_name()}")
        logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        default_batch_size = 4
        default_num_workers = 0
        logger.info("Using CPU - training will be slow")
    
    # Apply batch size (command line > config > device default)
    batch_size = args.batch_size or config.training.stage1_batch_size
    if device.type == "mps" and batch_size > 16:
        logger.warning(f"Batch size {batch_size} may cause memory issues on MPS. Consider --batch_size 8")
    config.training.stage1_batch_size = args.batch_size or default_batch_size
    config.training.stage2_batch_size = args.batch_size or default_batch_size
    
    # Apply num_workers
    num_workers = args.num_workers if args.num_workers is not None else default_num_workers
    
    # Apply gradient accumulation
    config.training.gradient_accumulation = args.gradient_accumulation
    
    # Apply custom epoch counts
    if args.stage1_epochs is not None:
        config.training.stage1_epochs = args.stage1_epochs
        logger.info(f"Stage 1 epochs override: {args.stage1_epochs}")
    if args.stage2_epochs is not None:
        config.training.stage2_epochs = args.stage2_epochs
        logger.info(f"Stage 2 epochs override: {args.stage2_epochs}")
    
    if args.log_dir:
        config.log_dir = args.log_dir
    
    # Create tokenizer
    tokenizer = TextTokenizer(vocab_size=config.text_vocab_size)
    logger.info(f"Tokenizer vocabulary size: {len(tokenizer)}")
    
    # Create model
    model = StreamingTTS(config)
    param_count = model.count_parameters()
    logger.info(f"Model parameters:")
    logger.info(f"  Main Transformer: {param_count['main_transformer']:,}")
    logger.info(f"  Depth Transformer: {param_count['depth_transformer']:,}")
    logger.info(f"  Total: {param_count['total']:,}")
    
    # Create dataloaders
    pin_memory = device.type == "cuda"
    
    train_dataloader, val_dataloader = create_train_val_dataloaders(
        data_dir=args.data_dir,
        tokenizer=tokenizer,
        batch_size=config.training.stage1_batch_size,
        num_workers=num_workers,
        num_codebooks=config.num_codebooks,
        pin_memory=pin_memory,
    )
    
    logger.info(f"Batch size: {config.training.stage1_batch_size}")
    logger.info(f"Gradient accumulation: {config.training.gradient_accumulation}")
    logger.info(f"Effective batch size: {config.training.stage1_batch_size * config.training.gradient_accumulation}")
    logger.info(f"Train batches: {len(train_dataloader)}")
    if val_dataloader:
        logger.info(f"Val batches: {len(val_dataloader)}")
    
    # Create trainer
    trainer = Trainer(
        config=config,
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        device=device,
        freeze_main_transformer=args.freeze_main_transformer,
    )
    
    # Handle checkpoint loading
    if args.resume:
        # Resume training - load full state (model + training progress)
        trainer.load_checkpoint(args.resume, weights_only=False)
        logger.info("Resuming training from checkpoint")
    elif args.checkpoint:
        # Load model weights only (for starting stage 2 from stage 1 checkpoint)
        trainer.load_checkpoint(args.checkpoint, weights_only=True)
        logger.info("Loaded model weights from checkpoint")
    
    # Validate start_stage with checkpoint
    if args.start_stage == 2 and not (args.checkpoint or args.resume):
        logger.warning("Starting at Stage 2 without a checkpoint - main transformer is untrained!")
    
    # Log training configuration
    logger.info(f"Starting at Stage: {args.start_stage}")
    if args.freeze_main_transformer:
        logger.info("Main transformer: FROZEN in Stage 2")
    else:
        logger.info("Main transformer: TRAINABLE in Stage 2")
    
    # Train
    trainer.train(start_stage=args.start_stage)


if __name__ == "__main__":
    main()
