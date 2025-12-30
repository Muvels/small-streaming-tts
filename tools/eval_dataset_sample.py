#!/usr/bin/env python3
"""
Eval/debug helper: compare ground-truth vs model-generated Mimi tokens/audio for one dataset sample.

Usage (run from repo root):
  ./.venv/bin/python tools/eval_dataset_sample.py \
    --checkpoint checkpoints/final.pt \
    --data_dir data/prepared_c4 \
    --manifest val.jsonl \
    --out_dir debug_out \
    --idx random \
    --device cpu

This writes:
  - gt.wav                 (decode of ground-truth tokens)
  - gen.wav                (decode of model-generated tokens)
  - sample.json            (text + metadata)
  - token_stats.txt        (CB1 accuracy vs GT for teacher-forced + generated)
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from small_tts.config import TTSConfig
from small_tts.model import StreamingTTS
from small_tts.data import TextTokenizer, PreEncodedTTSDataset


def load_config_from_checkpoint(ckpt: dict) -> TTSConfig:
    config = TTSConfig()
    config_dict = ckpt.get("config", {})
    for key, value in config_dict.get("main", {}).items():
        if hasattr(config.main, key):
            setattr(config.main, key, value)
    for key, value in config_dict.get("depth", {}).items():
        if hasattr(config.depth, key):
            setattr(config.depth, key, value)
    for key, value in config_dict.items():
        if not isinstance(value, dict) and hasattr(config, key):
            setattr(config, key, value)
    return config


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--manifest", type=str, default="val.jsonl")
    ap.add_argument("--idx", type=str, default="random", help='Sample index or "random"')
    ap.add_argument("--out_dir", type=str, default="debug_out")
    ap.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"])
    ap.add_argument("--max_frames", type=int, default=200, help="Max frames to generate for gen.wav")
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--top_p", type=float, default=0.9)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.checkpoint, map_location=args.device)
    config = load_config_from_checkpoint(ckpt)

    model = StreamingTTS(config).to(args.device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    tokenizer = TextTokenizer(vocab_size=config.text_vocab_size)
    ds = PreEncodedTTSDataset(
        data_dir=args.data_dir,
        manifest_file=args.manifest,
        tokenizer=tokenizer,
        num_codebooks=config.num_codebooks,
    )

    if args.idx == "random":
        idx = random.randrange(len(ds))
    else:
        idx = int(args.idx)

    ex = ds[idx]
    while ex is None:
        idx = random.randrange(len(ds))
        ex = ds[idx]
    meta = ds.samples[idx]

    # --- Ground truth audio decode ---
    gt_tokens = ex["audio_tokens"].unsqueeze(0).to(args.device)  # [1,4,T]
    gt_wav = model.codec.decode(gt_tokens)[0, 0].detach().cpu().numpy()
    sf.write((out_dir / "gt.wav").as_posix(), gt_wav, config.codec.sample_rate)

    # --- Teacher-forced CB1 accuracy (no padding in single sample) ---
    text = meta.get("text", "")
    text_tokens = ex["text_tokens"].unsqueeze(0).to(args.device)
    speaker = ex["speaker_id"].unsqueeze(0).to(args.device)
    lang = ex["language_id"].unsqueeze(0).to(args.device)

    cb1 = gt_tokens[:, 0, :]  # [1,T]
    cb1_in = cb1[:, :-1]
    cb1_tgt = cb1[:, 1:]

    logits, hidden, _ = model.main_transformer(
        text_tokens=text_tokens,
        audio_tokens=cb1_in,
        speaker_id=speaker,
        language_id=lang,
    )
    audio_start = 1 + text_tokens.shape[1]
    pred = logits[:, audio_start:audio_start + cb1_tgt.shape[1], :]
    tf_loss = F.cross_entropy(pred.reshape(-1, config.audio_vocab_size), cb1_tgt.reshape(-1), reduction="mean")
    tf_acc = (pred.argmax(dim=-1) == cb1_tgt).float().mean()

    # --- Generated audio ---
    gen = []
    state = model.init_streaming(speaker, lang, device=args.device)
    state = model.stream_text(state, text_tokens)
    for _ in range(args.max_frames):
        chunk, state = model.generate_audio_frame(
            state,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )
        gen.append(chunk)
    gen_wav = torch.cat(gen, dim=-1)[0, 0].detach().cpu().numpy()
    sf.write((out_dir / "gen.wav").as_posix(), gen_wav, config.codec.sample_rate)

    # Save metadata
    (out_dir / "sample.json").write_text(
        json.dumps(
            {
                "idx": idx,
                "text": text,
                "speaker_id": int(ex["speaker_id"].item()),
                "language_id": int(ex["language_id"].item()),
                "language": meta.get("language"),
                "conversation_id": meta.get("conversation_id"),
                "segment_idx": meta.get("segment_idx"),
                "audio_tokens_path": meta.get("audio_tokens_path"),
                "gt_frames": int(gt_tokens.shape[-1]),
                "gen_frames": int(args.max_frames),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    (out_dir / "token_stats.txt").write_text(
        f"teacher_forced_cb1_loss: {float(tf_loss):.4f}\n"
        f"teacher_forced_cb1_acc:  {float(tf_acc):.4f}\n",
        encoding="utf-8",
    )

    print("Wrote:", out_dir)
    print("  gt.wav, gen.wav, sample.json, token_stats.txt")


if __name__ == "__main__":
    main()


