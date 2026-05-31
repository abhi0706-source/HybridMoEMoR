#!/usr/bin/env python3
"""
HybridMoRMoE Full Training Pipeline — P100 x1 (SINGLE GPU, 16 GB)
==================================================================
Optimised for NVIDIA P100 (Pascal, compute 6.0, FP16, 16 GB HBM2).
KEY CHANGES vs T4×2 version
────────────────────────────
  • Single-GPU path only — no DataParallel / multi-GPU branching.
  • FP16 forced everywhere (P100 has NO BF16 support).
  • Batch size 2 + grad-accum 8 → eff batch 16 (P100 bandwidth > T4).
  • packing=True for SFT & pretrain → ~2× throughput on long-tail data.
  • Data volumes doubled:
        pretrain_max_samples  200 K → 400 K
        sft_max_samples/dom   5 K  → 10 K
        grpo_max_dataset      10 K → 20 K
  • dataloader_num_workers 2 → 4 (P100 hosts usually have ≥4 cores).
  • Save / eval frequency reduced to cut I/O overhead.
  • Sequence length stays 4096; RotaryEmbedding cache 8192.
  • OOM fallback in GRPO is more aggressive (batch 1, accum 16).
"""
import gc
import inspect
import json
import logging
import math
import os
import re
import shutil
import sys
import time
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional
# ── Unbuffered I/O ──
os.environ["PYTHONUNBUFFERED"] = "1"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN", "")  # Set via environment variable
os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "disabled"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
warnings.filterwarnings("ignore")
os.environ["TQDM_DISABLE"] = "1"
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationMixin,
    PretrainedConfig,
    PreTrainedModel,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
import trl
from trl import GRPOConfig, GRPOTrainer, SFTConfig, SFTTrainer
from datasets import Dataset, load_dataset
def _ensure_clean_distributed_state():
    try:
        if torch.distributed.is_initialized():
            try:
                torch.distributed.get_world_size()
                return
            except (ValueError, RuntimeError):
                try:
                    torch.distributed.destroy_process_group()
                except Exception:
                    pass
    except Exception:
        pass
    try:
        from accelerate.state import PartialState
        if hasattr(PartialState, '_shared_state') and PartialState._shared_state:
            PartialState._shared_state.clear()
    except Exception:
        pass
_ensure_clean_distributed_state()
IS_KAGGLE = os.path.exists("/kaggle")
_log_dir = "/kaggle/working/hybrid_mor_moe_P100" if IS_KAGGLE else "./hybrid_mor_moe_P100"
os.makedirs(_log_dir, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(_log_dir, "pipeline.log"), mode="a", encoding="utf-8"),
    ],
)
logger = logging.getLogger("HybridMoRMoE_P100")
OUTPUT_STORAGE_LIMIT_GB    = 12.0
OUTPUT_STORAGE_WARN_GB     =  9.5
OUTPUT_STORAGE_CRITICAL_GB = 11.0
# ════════════════════════════════════════════════════════════════════════════
#  Storage helpers (unchanged logic, lighter logging)
# ════════════════════════════════════════════════════════════════════════════
def get_dir_size_gb(path):
    if not os.path.isdir(path):
        return 0.0
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            try:
                total += os.path.getsize(os.path.join(dirpath, f))
            except OSError:
                pass
    return total / (1024 ** 3)
def check_output_storage(output_dir) -> str:
    used = get_dir_size_gb("/kaggle/working" if IS_KAGGLE else output_dir)
    pct = used / OUTPUT_STORAGE_LIMIT_GB * 100
    logger.info(f"  Storage: {used:.2f} GB / {OUTPUT_STORAGE_LIMIT_GB:.0f} GB  ({pct:.0f}%)")
    if used >= OUTPUT_STORAGE_CRITICAL_GB:
        logger.warning(f"  !! CRITICAL: {used:.2f} GB !!")
        return "critical"
    if used >= OUTPUT_STORAGE_WARN_GB:
        logger.warning(f"  !  WARNING : {used:.2f} GB")
        return "warn"
    return "ok"
def _rmdir(path: str, reason: str = ""):
    before = get_dir_size_gb(path)
    shutil.rmtree(path, ignore_errors=True)
    tag = f"  [{reason}]" if reason else ""
    logger.info(f"  Removed{tag}: {path}  (freed ~{before:.2f} GB)")
def emergency_cleanup(output_dir: str, level: str = "warn"):
    base = "/kaggle/working" if IS_KAGGLE else output_dir
    def _used():
        return get_dir_size_gb(base)
    logger.info(f"  [Cleanup/{level}] Starting — current usage {_used():.2f} GB")
    for subdir in ["pretrain", "sft", "grpo"]:
        phase_dir = os.path.join(output_dir, subdir)
        if not os.path.isdir(phase_dir):
            continue
        ckpts = sorted(
            [d for d in os.listdir(phase_dir)
             if d.startswith("checkpoint-") and os.path.isdir(os.path.join(phase_dir, d))],
            key=lambda x: int(x.split("-")[-1]),
        )
        for ckpt in ckpts[:-1]:
            _rmdir(os.path.join(phase_dir, ckpt), "T1-old-ckpt")
        if level == "critical" and ckpts:
            _rmdir(os.path.join(phase_dir, ckpts[-1]), "T1-latest-ckpt")
    if _used() < OUTPUT_STORAGE_WARN_GB:
        return "ok"
    sft_done  = os.path.isdir(os.path.join(output_dir, "sft_model"))
    grpo_done = os.path.isdir(os.path.join(output_dir, "final_model"))
    if os.path.isdir(p := os.path.join(output_dir, "pretrain_model")) and sft_done:
        _rmdir(p, "T2-pretrain_model")
    if os.path.isdir(p := os.path.join(output_dir, "sft_model")) and grpo_done:
        _rmdir(p, "T2-sft_model")
    if _used() < OUTPUT_STORAGE_WARN_GB:
        return "ok"
    if os.path.isdir(p := os.path.join(output_dir, "best_pretrain")):
        _rmdir(p, "T3-best_pretrain")
    used_after = _used()
    return "ok" if used_after < OUTPUT_STORAGE_WARN_GB else (
        "critical" if used_after >= OUTPUT_STORAGE_CRITICAL_GB else "warn")
def enforce_storage_limit(output_dir: str, action: str = "save"):
    used = get_dir_size_gb("/kaggle/working" if IS_KAGGLE else output_dir)
    if used >= OUTPUT_STORAGE_LIMIT_GB:
        status = emergency_cleanup(output_dir, level="critical")
        used_after = get_dir_size_gb("/kaggle/working" if IS_KAGGLE else output_dir)
        if used_after >= OUTPUT_STORAGE_LIMIT_GB:
            raise RuntimeError(f"[StorageGate] Cannot {action}: {used_after:.2f} GB used")
    elif used >= OUTPUT_STORAGE_CRITICAL_GB:
        emergency_cleanup(output_dir, level="critical")
    elif used >= OUTPUT_STORAGE_WARN_GB:
        emergency_cleanup(output_dir, level="warn")
# ════════════════════════════════════════════════════════════════════════════
#  GPU setup — single P100 path
# ════════════════════════════════════════════════════════════════════════════
def setup_gpu():
    if not torch.cuda.is_available():
        logger.warning("No CUDA device. Running on CPU.")
        return False, 0
    num_gpus = torch.cuda.device_count()
    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        gpu_name = props.name
        vram_gb = props.total_memory / 1e9
        cc = torch.cuda.get_device_capability(i)
        logger.info(f"GPU {i}: {gpu_name} | VRAM: {vram_gb:.1f} GB | Compute: {cc[0]}.{cc[1]}")
    # P100 = compute 6.0, NO BF16, good FP16 throughput
    torch.backends.cuda.matmul.allow_tf32 = False   # Pascal has no TF32
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = True
    logger.info(f"Precision: FP16 (P100 — no BF16) | GPUs visible: {num_gpus}")
    torch.cuda.set_per_process_memory_fraction(0.95, 0)
    torch.cuda.empty_cache()
    gc.collect()
    return True, num_gpus
# ════════════════════════════════════════════════════════════════════════════
#  Model presets
# ════════════════════════════════════════════════════════════════════════════
MODEL_PRESETS = {
    "small": {
        "d_model": 512, "n_heads": 8, "d_ff": 1408,
        "num_base_layers": 4, "num_shared_blocks": 3,
        "num_recursions": 2, "num_unique_last_layers": 1,
        "num_experts": 4, "max_recursions": 2,
    },
    "medium": {
        "d_model": 576, "n_heads": 8, "d_ff": 1536,
        "num_base_layers": 6, "num_shared_blocks": 6,
        "num_recursions": 3, "num_unique_last_layers": 2,
        "num_experts": 4, "max_recursions": 3,
    },
    "large": {
        "d_model": 1536, "n_heads": 16, "d_ff": 4096,
        "num_base_layers": 8, "num_shared_blocks": 8,
        "num_recursions": 3, "num_unique_last_layers": 3,
        "num_experts": 8, "max_recursions": 3,
    },
}
# ════════════════════════════════════════════════════════════════════════════
#  Pipeline config — P100 optimised defaults
# ════════════════════════════════════════════════════════════════════════════
@dataclass
class PipelineConfig:
    model_size: str = "medium"
    max_seq_len: int = 4096
    dropout: float = 0.05
    num_gpus: int = 1                # ← single P100
    sft_data_dir: str = "/kaggle/input/datasets/abhishek0706/sft-dataset"
    pretrain_corpus: str = "./pretraining_corpus.jsonl"
    tokenizer_path: str = "./hf_assets/tokenizer/Qwen2.5-0.5B-Instruct"
    tokenizer_hf_id: str = "Qwen/Qwen2.5-0.5B-Instruct"
    output_dir: str = "./hybrid_mor_moe_P100"
    # ── Pretrain (doubled data) ──
    pretrain_max_samples: int = 400_000       # was 200 K
    pretrain_max_steps: int = 10_000
    pretrain_batch_size: int = 2              # P100 16 GB can handle bs=2 @ 4096
    pretrain_grad_accum: int = 8              # eff batch = 16
    pretrain_lr: float = 3e-4
    pretrain_warmup_steps: int = 500
    pretrain_weight_decay: float = 0.1
    pretrain_save_steps: int = 2500           # save less often → faster
    pretrain_eval_steps: int = 2500
    pretrain_logging_steps: int = 50
    pretrain_eval_split: float = 0.02
    # ── SFT (doubled data, packing ON) ──
    sft_max_samples_per_domain: int = 10_000  # was 5 K
    sft_max_steps: int = 2000
    sft_batch_size: int = 2
    sft_grad_accum: int = 8                   # eff batch = 16
    sft_lr: float = 5e-4
    sft_warmup_steps: int = 200
    sft_weight_decay: float = 0.1
    sft_max_grad_norm: float = 1.0
    sft_save_steps: int = 1000
    sft_eval_steps: int = 500
    sft_logging_steps: int = 25
    sft_eval_split: float = 0.05
    # ── GRPO (doubled data) ──
    grpo_max_steps: int = 1000
    grpo_batch_size: int = 2
    grpo_grad_accum: int = 8                  # eff batch = 16
    grpo_lr: float = 5e-6
    grpo_warmup_steps: int = 50
    grpo_weight_decay: float = 0.05
    grpo_max_grad_norm: float = 0.5
    grpo_num_generations: int = 2
    grpo_max_completion_length: int = 192
    grpo_max_prompt_length: int = 128
    grpo_beta: float = 0.04
    grpo_save_steps: int = 500
    grpo_logging_steps: int = 25
    grpo_max_dataset_size: int = 20_000       # was 10 K
    save_total_limit: int = 2
    dataloader_num_workers: int = 4           # was 2
    inference_every_steps: int = 1000
    skip_pretrain: bool = True
    skip_sft: bool = True
def adjust_config_for_model_size(cfg: PipelineConfig):
    """Tune batch / seq sizes per model preset for P100 16 GB."""
    if cfg.model_size == "large":
        cfg.max_seq_len = 512
        cfg.pretrain_batch_size = 1
        cfg.pretrain_grad_accum = 16
        cfg.sft_batch_size = 1
        cfg.sft_grad_accum = 16
        cfg.grpo_batch_size = 1
        cfg.grpo_grad_accum = 16
        cfg.grpo_num_generations = 2
        cfg.grpo_max_completion_length = 256
        cfg.grpo_max_prompt_length = 256
    elif cfg.model_size == "medium":
        # P100 16 GB VRAM budget for GRPO (294M model):
        #   Model FP16:       ~600 MB
        #   Optimizer FP32:   ~2.4 GB
        #   Gradients:        ~600 MB
        #   Base overhead:    ~3.6 GB → leaves ~12 GB for activations + logits
        #
        # GRPO scoring forward pass (with grads) over batch × seq × 151K vocab is
        # the bottleneck.  Accelerate's convert_to_fp32 doubles logits memory.
        # Keep total tokens LOW: prompt=128 + completion=192 = 320 total.
        cfg.max_seq_len = 4096
        cfg.pretrain_batch_size = 2
        cfg.pretrain_grad_accum = 8
        cfg.sft_batch_size = 2
        cfg.sft_grad_accum = 8
        cfg.grpo_batch_size = 1
        cfg.grpo_grad_accum = 16              # eff batch = 16
        cfg.grpo_num_generations = 2
        cfg.grpo_max_completion_length = 192  # conservative: 128+192=320 total
        cfg.grpo_max_prompt_length = 128
    else:  # small — more room, but 152K vocab still limits GRPO
        cfg.max_seq_len = 4096
        cfg.pretrain_batch_size = 4
        cfg.pretrain_grad_accum = 4
        cfg.sft_batch_size = 4
        cfg.sft_grad_accum = 4
        cfg.grpo_batch_size = 1
        cfg.grpo_grad_accum = 16
        cfg.grpo_max_completion_length = 256
        cfg.grpo_max_prompt_length = 192
    eff_sft  = cfg.sft_batch_size  * cfg.sft_grad_accum
    eff_grpo = cfg.grpo_batch_size * cfg.grpo_grad_accum
    logger.info(f"P100 config — {cfg.model_size} model: seq={cfg.max_seq_len}")
    logger.info(f"  Per-device batch : SFT={cfg.sft_batch_size}, GRPO={cfg.grpo_batch_size}")
    logger.info(f"  Grad accum       : SFT={cfg.sft_grad_accum}, GRPO={cfg.grpo_grad_accum}")
    logger.info(f"  Effective batch  : SFT={eff_sft}, GRPO={eff_grpo}")
    logger.info(f"  GRPO seq lengths : prompt={cfg.grpo_max_prompt_length}, "
                f"completion={cfg.grpo_max_completion_length}")
    return cfg
# ════════════════════════════════════════════════════════════════════════════
#  Model Architecture  (identical to original — kept for self-containedness)
# ════════════════════════════════════════════════════════════════════════════
class HybridMoRMoEConfig(PretrainedConfig):
    model_type = "hybrid_mor_moe"
    model_size: str = "medium"
    d_model: int = 576
    n_heads: int = 8
    d_ff: int = 1536
    vocab_size: int = 151936
    max_seq_len: int = 4096
    dropout: float = 0.05
    num_base_layers: int = 4
    num_shared_blocks: int = 4
    num_recursions: int = 2
    max_recursions: int = 2
    num_unique_last_layers: int = 2
    router_percentile: float = 0.7
    num_experts: int = 4
    num_experts_per_tok: int = 1
    router_aux_loss_coef: float = 0.0001
    moe_aux_loss_coef: float = 0.0001
    complexity_hidden_dim: int = 64
    complexity_threshold_easy: float = 0.3
    complexity_threshold_hard: float = 0.7
    think_budget_easy: int = 12
    think_budget_medium: int = 48
    think_budget_hard: int = 96
    def __init__(self, **kwargs):
        model_size = kwargs.get("model_size", "small")
        if model_size in MODEL_PRESETS:
            for k, v in MODEL_PRESETS[model_size].items():
                if k not in kwargs:
                    kwargs[k] = v
        super().__init__(**kwargs)
        self.model_size = model_size
        n_rec = min(self.num_recursions, self.max_recursions)
        self.num_hidden_layers = (
            self.num_base_layers
            + n_rec * self.num_shared_blocks
            + n_rec * self.num_unique_last_layers
        )
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=8192):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self._set_cos_sin_cache(max_seq_len)
    def _set_cos_sin_cache(self, seq_len):
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)
        self.max_seq_len_cached = seq_len
    def forward(self, seq_len, device):
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len)
        return self.cos_cached[:seq_len].to(device), self.sin_cached[:seq_len].to(device)
def apply_rotary_emb(q, k, cos, sin):
    def rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    seq_len = q.shape[2]
    cos = cos[:seq_len].unsqueeze(0).unsqueeze(0)
    sin = sin[:seq_len].unsqueeze(0).unsqueeze(0)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.05):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.d_model = d_model
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_dropout_p = dropout
    def forward(self, x, mask=None, cos=None, sin=None, past_key_value=None, use_cache=False):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        if cos is not None and sin is not None:
            q, k = apply_rotary_emb(q, k, cos, sin)
        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        new_cache = (k, v) if use_cache else None
        dropout_p = self.attn_dropout_p if self.training else 0.0
        attn_out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=dropout_p,
            is_causal=(past_key_value is None),
        )
        output = self.o_proj(attn_out.transpose(1, 2).contiguous().view(B, T, C))
        return output, new_cache
class Expert(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.05):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        return self.w2(self.dropout(F.silu(self.w1(x)) * self.w3(x)))
class MoELayer(nn.Module):
    def __init__(self, d_model, d_ff, num_experts, top_k, dropout=0.05):
        super().__init__()
        self.num_experts, self.top_k = num_experts, top_k
        self.experts = nn.ModuleList([Expert(d_model, d_ff, dropout) for _ in range(num_experts)])
        self.gate = nn.Linear(d_model, num_experts, bias=False)
    def forward(self, x):
        B, T, C = x.shape
        xf = x.reshape(-1, C)
        gp = F.softmax(self.gate(xf), dim=-1)
        tp, ti = torch.topk(gp, self.top_k, dim=-1)
        tp = tp / (tp.sum(dim=-1, keepdim=True) + 1e-8)
        out = torch.zeros_like(xf)
        for i in range(self.num_experts):
            m = (ti == i).any(dim=-1)
            if m.any():
                eo = self.experts[i](xf[m])
                w = (tp[m] * (ti[m] == i).float()).sum(dim=-1, keepdim=True)
                out[m] += w * eo
        aux_loss = (gp.mean(0) ** 2).sum() * self.num_experts
        return out.view(B, T, C), aux_loss
class PercentileRouter(nn.Module):
    def __init__(self, d_model, percentile=0.7):
        super().__init__()
        self.percentile = percentile
        self.router = nn.Linear(d_model, 1)
    def forward(self, x, mask=None):
        device = x.device
        raw = self.router(x).squeeze(-1).clamp(-50.0, 50.0)
        scores = torch.softmax(raw, dim=-1)
        if torch.isnan(scores).any() or torch.isinf(scores).any():
            scores = torch.where(
                torch.isnan(scores) | torch.isinf(scores),
                torch.ones_like(scores) / max(scores.shape[-1], 1), scores,
            )
        if mask is not None:
            am = mask.bool().to(device)
            if am.shape != scores.shape:
                if am.shape[0] == scores.shape[0] and am.shape[-1] >= scores.shape[-1]:
                    am = am[..., -scores.shape[-1]:]
                else:
                    am = torch.ones_like(scores, dtype=torch.bool, device=device)
        else:
            am = torch.ones_like(scores, dtype=torch.bool, device=device)
        active = scores[am]
        if active.numel() > 0:
            thr = torch.quantile(active.float(), self.percentile)
            sel = (scores >= thr) & am
        else:
            sel = am
        zl = torch.logsumexp(scores[am].float(), dim=0) ** 2 if am.any() else torch.tensor(0.0, device=device)
        return sel, scores, zl
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout, use_moe=False, num_experts=8, top_k=2):
        super().__init__()
        self.use_moe = use_moe
        self.ln1 = nn.RMSNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ln2 = nn.RMSNorm(d_model)
        if use_moe:
            self.ffn = MoELayer(d_model, d_ff, num_experts, top_k, dropout)
        else:
            self.w1 = nn.Linear(d_model, d_ff, bias=False)
            self.w3 = nn.Linear(d_model, d_ff, bias=False)
            self.w2 = nn.Linear(d_ff, d_model, bias=False)
            self.ffn_dropout = nn.Dropout(dropout)
    def _ffn(self, x):
        if self.use_moe:
            return self.ffn(self.ln2(x))
        else:
            h = self.ln2(x)
            return self.w2(self.ffn_dropout(F.silu(self.w1(h)) * self.w3(h))), None
    def forward(self, x, mask=None, cos=None, sin=None, past_key_value=None, use_cache=False):
        attn_out, new_cache = self.attn(self.ln1(x), mask, cos, sin, past_key_value, use_cache)
        x = x + attn_out
        fo, ml = self._ffn(x)
        return x + fo, ml, new_cache
class ComplexityScorer(nn.Module):
    def __init__(self, d_model, hidden_dim=128):
        super().__init__()
        self.pool_proj = nn.Linear(d_model, hidden_dim)
        self.scorer = nn.Sequential(
            nn.RMSNorm(hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
    def forward(self, hidden_states, attention_mask=None):
        if attention_mask is not None:
            m = attention_mask.unsqueeze(-1).float()
            pooled = (hidden_states * m).sum(1) / m.sum(1).clamp(min=1)
        else:
            pooled = hidden_states.mean(dim=1)
        return torch.sigmoid(self.scorer(self.pool_proj(pooled)).squeeze(-1))
class HybridMoRMoEForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = HybridMoRMoEConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _supports_sdpa = False
    _no_split_modules = []
    @classmethod
    def _can_set_experts_implementation(cls) -> bool:
        return False
    def __init__(self, config: HybridMoRMoEConfig):
        super().__init__(config)
        self.config = config
        self.gradient_checkpointing = False
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.rotary_emb = RotaryEmbedding(config.d_model // config.n_heads, config.max_seq_len * 2)
        self.base_layers = nn.ModuleList([
            TransformerBlock(config.d_model, config.n_heads, config.d_ff, config.dropout, False)
            for _ in range(config.num_base_layers)
        ])
        self.shared_blocks = nn.ModuleList([
            TransformerBlock(config.d_model, config.n_heads, config.d_ff, config.dropout,
                             True, config.num_experts, config.num_experts_per_tok)
            for _ in range(config.num_shared_blocks)
        ])
        self.routers = nn.ModuleList([
            PercentileRouter(config.d_model, config.router_percentile)
            for _ in range(config.num_recursions)
        ])
        self.unique_last_layers = nn.ModuleList([
            nn.ModuleList([
                TransformerBlock(config.d_model, config.n_heads, config.d_ff, config.dropout, False)
                for _ in range(config.num_unique_last_layers)
            ])
            for _ in range(config.num_recursions)
        ])
        self.complexity_scorer = ComplexityScorer(config.d_model, config.complexity_hidden_dim)
        self.ln_f = nn.RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self._num_kv_layers = self._count_kv_layers()
        self.post_init()
    def _count_kv_layers(self):
        count = len(self.base_layers)
        n_rec = min(self.config.num_recursions, len(self.routers))
        for ri in range(n_rec):
            count += len(self.shared_blocks) + len(self.unique_last_layers[ri])
        return count
    def _set_gradient_checkpointing(self, enable=True, gradient_checkpointing_func=None):
        self.gradient_checkpointing = enable
    def _init_weights(self, module):
        std = 0.02
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, std=std)
    def get_input_embeddings(self):
        return self.token_embedding
    def set_input_embeddings(self, value):
        self.token_embedding = value
    def get_output_embeddings(self):
        return self.lm_head
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings
    def forward(self, input_ids=None, attention_mask=None, labels=None,
                past_key_values=None, use_cache=False, return_dict=True, **kwargs):
        from transformers.cache_utils import DynamicCache
        device = input_ids.device
        B, seq_len = input_ids.shape
        input_ids = input_ids.clamp(0, self.config.vocab_size - 1)
        x = self.token_embedding(input_ids)
        _input_is_dynamic_cache = isinstance(past_key_values, DynamicCache)
        if _input_is_dynamic_cache:
            if past_key_values.get_seq_length() > 0:
                past_key_values = [
                    (past_key_values.key_cache[i], past_key_values.value_cache[i])
                    for i in range(len(past_key_values.key_cache))
                ]
            else:
                past_key_values = None
        past_length = 0
        if (past_key_values is not None and isinstance(past_key_values, (list, tuple))
                and len(past_key_values) > 0 and past_key_values[0] is not None):
            past_length = past_key_values[0][0].shape[2]
        total_len = past_length + seq_len
        cos, sin = self.rotary_emb(total_len, device)
        cos = cos[past_length:total_len]
        sin = sin[past_length:total_len]
        new_past_key_values = []
        layer_idx = 0
        use_ckpt = self.gradient_checkpointing and self.training and not use_cache
        for layer in self.base_layers:
            past_kv = past_key_values[layer_idx] if past_key_values and layer_idx < len(past_key_values) else None
            if use_ckpt:
                x, _, new_cache = torch.utils.checkpoint.checkpoint(
                    layer, x, attention_mask, cos, sin, past_kv, use_cache, use_reentrant=False)
            else:
                x, _, new_cache = layer(x, attention_mask, cos, sin, past_kv, use_cache)
            new_past_key_values.append(new_cache)
            layer_idx += 1
        router_losses, moe_losses = [], []
        n_rec = min(self.config.num_recursions, len(self.routers))
        for ri in range(n_rec):
            sel, _, zl = self.routers[ri](x, attention_mask)
            router_losses.append(zl)
            for blk in self.shared_blocks:
                past_kv = past_key_values[layer_idx] if past_key_values and layer_idx < len(past_key_values) else None
                if use_ckpt:
                    xb, ml, new_cache = torch.utils.checkpoint.checkpoint(
                        blk, x, attention_mask, cos, sin, past_kv, use_cache, use_reentrant=False)
                else:
                    xb, ml, new_cache = blk(x, attention_mask, cos, sin, past_kv, use_cache)
                x = torch.where(sel.unsqueeze(-1), xb, x)
                new_past_key_values.append(new_cache)
                layer_idx += 1
                if ml is not None:
                    moe_losses.append(ml)
            for layer in self.unique_last_layers[ri]:
                past_kv = past_key_values[layer_idx] if past_key_values and layer_idx < len(past_key_values) else None
                if use_ckpt:
                    x, _, new_cache = torch.utils.checkpoint.checkpoint(
                        layer, x, attention_mask, cos, sin, past_kv, use_cache, use_reentrant=False)
                else:
                    x, _, new_cache = layer(x, attention_mask, cos, sin, past_kv, use_cache)
                new_past_key_values.append(new_cache)
                layer_idx += 1
        x = self.ln_f(x)
        logits = self.lm_head(x)
        # In-place cleanup — avoids allocating copies of the huge logits tensor
        logits.nan_to_num_(nan=0.0, posinf=100.0, neginf=-100.0)
        logits.clamp_(-100.0, 100.0)
        loss = None
        if labels is not None:
            cl = labels.clone()
            v = cl != -100
            cl[v] = cl[v].clamp(0, self.config.vocab_size - 1)
            sl = logits[..., :-1, :].contiguous()
            tl = cl[..., 1:].contiguous()
            loss = F.cross_entropy(sl.view(-1, sl.size(-1)), tl.view(-1), ignore_index=-100)
            if router_losses:
                loss = loss + self.config.router_aux_loss_coef * torch.stack(router_losses).mean()
            if moe_losses:
                loss = loss + self.config.moe_aux_loss_coef * torch.stack(moe_losses).mean()
        output_cache = tuple(new_past_key_values) if use_cache else None
        if return_dict:
            from transformers.modeling_outputs import CausalLMOutputWithPast
            return CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=output_cache)
        return (loss, logits, output_cache) if loss is not None else (logits, output_cache)
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, **kwargs):
        from transformers.cache_utils import DynamicCache
        has_past = False
        if past_key_values is not None:
            if isinstance(past_key_values, DynamicCache):
                has_past = past_key_values.get_seq_length() > 0
            elif isinstance(past_key_values, (list, tuple)) and len(past_key_values) > 0:
                has_past = past_key_values[0] is not None
        if has_past:
            input_ids = input_ids[:, -1:]
        return {"input_ids": input_ids, "attention_mask": attention_mask,
                "past_key_values": past_key_values, "use_cache": True}
    @torch.no_grad()
    def simple_generate(self, input_ids, max_new_tokens=256, temperature=0.7,
                        top_k=50, top_p=0.9, pad_token_id=0, eos_token_id=None, use_cache=True):
        self.eval()
        gen_model = self.module if hasattr(self, 'module') else self
        generated = input_ids.clone()
        past_key_values = None
        for _ in range(max_new_tokens):
            current_input = generated[:, -1:] if (past_key_values is not None and use_cache) else generated
            outputs = gen_model.forward(current_input, past_key_values=past_key_values,
                                        use_cache=use_cache, return_dict=True)
            if use_cache:
                past_key_values = outputs.past_key_values
            next_logits = outputs.logits[:, -1, :].float() / max(temperature, 1e-8)
            if top_k > 0:
                v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits[next_logits < v[..., -1, None]] = float("-inf")
            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(next_logits, descending=True)
                cumsum = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                remove = cumsum > top_p
                remove[..., 1:] = remove[..., :-1].clone()
                remove[..., 0] = 0
                next_logits[remove.scatter(1, sorted_idx, remove)] = float("-inf")
            probs = F.softmax(next_logits, dim=-1).clamp(min=0.0)
            if torch.isnan(probs).any() or probs.sum(dim=-1).min() < 1e-8:
                probs = torch.ones_like(probs) / probs.shape[-1]
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break
        self.train()
        return generated
AutoConfig.register("hybrid_mor_moe", HybridMoRMoEConfig)
AutoModelForCausalLM.register(HybridMoRMoEConfig, HybridMoRMoEForCausalLM)
setattr(transformers, "HybridMoRMoEForCausalLM", HybridMoRMoEForCausalLM)
# ════════════════════════════════════════════════════════════════════════════
#  Checkpoint Utilities
# ════════════════════════════════════════════════════════════════════════════
def find_latest_checkpoint(output_dir):
    if not os.path.isdir(output_dir):
        return None
    checkpoints = [d for d in os.listdir(output_dir)
                   if d.startswith("checkpoint-") and os.path.isdir(os.path.join(output_dir, d))]
    if not checkpoints:
        return None
    checkpoints.sort(key=lambda x: int(x.split("-")[-1]))
    latest = os.path.join(output_dir, checkpoints[-1])
    logger.info(f"  Found checkpoint to resume from: {latest}")
    return latest
def cleanup_checkpoints(output_dir, keep_last=0):
    if not os.path.isdir(output_dir):
        return
    checkpoints = sorted(
        [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")
         and os.path.isdir(os.path.join(output_dir, d))],
        key=lambda x: int(x.split("-")[-1]),
    )
    to_remove = checkpoints[:-keep_last] if keep_last > 0 else checkpoints
    for ckpt in to_remove:
        path = os.path.join(output_dir, ckpt)
        shutil.rmtree(path, ignore_errors=True)
        logger.info(f"  Cleaned up checkpoint: {path}")
# ════════════════════════════════════════════════════════════════════════════
#  Robust Model Loading
# ════════════════════════════════════════════════════════════════════════════
def load_checkpoint_robust(config, checkpoint_dir, device="cpu"):
    from safetensors.torch import load_file as safetensors_load
    model = HybridMoRMoEForCausalLM(config)
    if not os.path.isdir(checkpoint_dir):
        raise FileNotFoundError(f"Checkpoint dir not found: {checkpoint_dir}")
    sf_files = sorted(f for f in os.listdir(checkpoint_dir) if f.endswith(".safetensors"))
    if sf_files:
        ckpt_state = {}
        for sf in sf_files:
            ckpt_state.update(safetensors_load(os.path.join(checkpoint_dir, sf), device="cpu"))
    else:
        pt_bin = os.path.join(checkpoint_dir, "pytorch_model.bin")
        if not os.path.isfile(pt_bin):
            raise FileNotFoundError(f"No .safetensors or pytorch_model.bin in {checkpoint_dir}")
        ckpt_state = torch.load(pt_bin, map_location="cpu", weights_only=False)
    model_state = model.state_dict()
    loaded, skipped_unexpected, partial_loaded = 0, 0, 0
    for key, ckpt_param in ckpt_state.items():
        if key not in model_state:
            skipped_unexpected += 1
            continue
        model_param = model_state[key]
        if ckpt_param.shape == model_param.shape:
            model_state[key] = ckpt_param
            loaded += 1
        else:
            slices = tuple(
                slice(0, min(cs, ms))
                for cs, ms in zip(ckpt_param.shape, model_param.shape)
            )
            model_state[key][slices] = ckpt_param[slices]
            partial_loaded += 1
            logger.info(f"  [load] Partial copy {key}: ckpt={list(ckpt_param.shape)} → model={list(model_param.shape)}")
    missing = [k for k in model_state if k not in ckpt_state]
    model.load_state_dict(model_state, strict=True)
    logger.info(f"  [load] Loaded: {loaded} | Partial: {partial_loaded} | "
                f"Unexpected (skipped): {skipped_unexpected} | Missing (random init): {len(missing)}")
    model.to(device)
    return model
# ════════════════════════════════════════════════════════════════════════════
#  Pipeline State Manager
# ════════════════════════════════════════════════════════════════════════════
class PipelineStateManager:
    def __init__(self, output_dir: str):
        self.path = os.path.join(output_dir, "pipeline_state.json")
        self._state = self._load()
    def _load(self) -> dict:
        if os.path.exists(self.path):
            try:
                with open(self.path, "r") as f:
                    state = json.load(f)
                logger.info(f"  [Checkpoint] Loaded pipeline state: completed={state.get('completed_phases', [])}")
                return state
            except Exception as e:
                logger.warning(f"  [Checkpoint] Could not read pipeline_state.json: {e}")
        return {"completed_phases": [], "best_eval_loss": {}, "phase_steps": {}}
    def _save(self):
        self._state["last_updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self._state, f, indent=2)
        os.replace(tmp, self.path)
    def mark_complete(self, phase: str, best_eval_loss: float = None, steps: int = None):
        if phase not in self._state["completed_phases"]:
            self._state["completed_phases"].append(phase)
        if best_eval_loss is not None:
            self._state["best_eval_loss"][phase] = round(best_eval_loss, 6)
        if steps is not None:
            self._state["phase_steps"][phase] = steps
        self._save()
        logger.info(f"  [Checkpoint] Phase '{phase}' marked complete")
    def is_complete(self, phase: str) -> bool:
        return phase in self._state["completed_phases"]
    def get_best_loss(self, phase: str) -> Optional[float]:
        return self._state["best_eval_loss"].get(phase)
    def summary(self) -> str:
        done = self._state.get("completed_phases", [])
        losses = self._state.get("best_eval_loss", {})
        parts = []
        for p in done:
            l = losses.get(p)
            parts.append(f"{p}(loss={l:.4f})" if l else p)
        return "Completed: " + (", ".join(parts) if parts else "none")
# ════════════════════════════════════════════════════════════════════════════
#  Callbacks
# ════════════════════════════════════════════════════════════════════════════
class BestModelCallback(TrainerCallback):
    def __init__(self, output_dir: str, phase: str, tokenizer):
        self.best_dir = os.path.join(output_dir, f"best_{phase}")
        self.phase = phase
        self.tokenizer = tokenizer
        self.best_loss = float("inf")
    def on_evaluate(self, args, state, control, model=None, metrics=None, **kwargs):
        if metrics is None or model is None:
            return
        eval_loss = metrics.get("eval_loss")
        if eval_loss is None:
            return
        if eval_loss < self.best_loss:
            self.best_loss = eval_loss
            save_model = model.module if hasattr(model, "module") else model
            save_model.save_pretrained(self.best_dir)
            self.tokenizer.save_pretrained(self.best_dir)
            meta = {"step": state.global_step, "eval_loss": round(eval_loss, 6),
                    "phase": self.phase, "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")}
            with open(os.path.join(self.best_dir, "best_checkpoint_meta.json"), "w") as f:
                json.dump(meta, f, indent=2)
            logger.info(f"  [BestModel/{self.phase}] New best eval_loss={eval_loss:.4f} @ step {state.global_step}")
class StorageMonitorCallback(TrainerCallback):
    def __init__(self, output_dir, check_every_steps=200):
        self.output_dir = output_dir
        self.check_every_steps = check_every_steps
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step <= 0 or state.global_step % self.check_every_steps != 0:
            return
        status = check_output_storage(self.output_dir)
        if status == "critical":
            result = emergency_cleanup(self.output_dir, level="critical")
            if result == "critical":
                control.should_training_stop = True
        elif status == "warn":
            emergency_cleanup(self.output_dir, level="warn")
class ValidationLoggerCallback(TrainerCallback):
    def __init__(self, phase=""):
        self.phase = phase
        self.eval_history = []
        self.best_eval_loss = float("inf")
        self.best_step = 0
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            return
        step = state.global_step
        eval_loss = metrics.get("eval_loss")
        if eval_loss is None:
            return
        self.eval_history.append((step, eval_loss))
        is_best = eval_loss < self.best_eval_loss
        if is_best:
            self.best_eval_loss = eval_loss
            self.best_step = step
        try:
            ppl_str = f" | ppl={math.exp(eval_loss):.2f}"
        except OverflowError:
            ppl_str = ""
        best_str = " BEST" if is_best else f" (best={self.best_eval_loss:.4f} @{self.best_step})"
        logger.info(f"  [{self.phase}] Step {step}: eval_loss={eval_loss:.4f}{ppl_str}{best_str}")
    def on_train_end(self, args, state, control, **kwargs):
        if self.eval_history:
            logger.info(f"  [{self.phase}] Summary: best={self.best_eval_loss:.4f} @step {self.best_step}")
class PrintProgressCallback(TrainerCallback):
    def __init__(self, phase=""):
        self.phase = phase
        self.start_time = None
    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        print(f"\n{'='*70}", flush=True)
        print(f"[{self.phase}] Training started | max_steps={args.max_steps} | "
              f"lr={args.learning_rate} | gpu=P100", flush=True)
        print(f"{'='*70}", flush=True)
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None or state.global_step == 0:
            return
        elapsed = time.time() - self.start_time
        steps_done = state.global_step
        steps_total = args.max_steps if args.max_steps > 0 else state.max_steps
        speed = steps_done / elapsed if elapsed > 0 else 0
        eta = (steps_total - steps_done) / speed if speed > 0 else 0
        loss = logs.get("loss", logs.get("train_loss"))
        lr = logs.get("learning_rate")
        parts = [f"[{self.phase}] step {steps_done}/{steps_total}"]
        if loss is not None: parts.append(f"loss={loss:.4f}")
        if lr is not None: parts.append(f"lr={lr:.2e}")
        parts.append(f"{speed:.2f} it/s")
        parts.append(f"eta={eta/60:.1f}m")
        print(" | ".join(parts), flush=True)
    def on_train_end(self, args, state, control, **kwargs):
        elapsed = time.time() - self.start_time
        print(f"[{self.phase}] Done: {state.global_step} steps in {elapsed/60:.1f}m", flush=True)
class PipelineCallback(TrainerCallback):
    def __init__(self, model, tokenizer, eval_prompts, phase="", eval_every=1000, max_new_tokens=256):
        self.model = model
        self.tokenizer = tokenizer
        self.eval_prompts = eval_prompts
        self.phase = phase
        self.eval_every = eval_every
        self.max_new_tokens = max_new_tokens
        self.start_time = None
    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step > 0 and state.global_step % 500 == 0 and torch.cuda.is_available():
            mem = torch.cuda.memory_allocated(0) / 1e9
            logger.info(f"  [{self.phase}] Step {state.global_step} | GPU 0: {mem:.1f}GB")
        if state.global_step > 0 and state.global_step % self.eval_every == 0:
            self._run_inference(state.global_step)
    def on_train_end(self, args, state, control, **kwargs):
        elapsed = time.time() - self.start_time
        logger.info(f"{self.phase} complete: {state.global_step} steps in {elapsed/3600:.2f}h")
        self._run_inference(state.global_step, final=True)
    @torch.no_grad()
    def _run_inference(self, step, final=False):
        self.model.eval()
        device = next(self.model.parameters()).device
        tag = "FINAL" if final else f"Step {step}"
        logger.info(f"\n--- {self.phase} Inference @ {tag} ---")
        prompts_dict = self.eval_prompts if isinstance(self.eval_prompts, dict) else {"general": self.eval_prompts}
        for domain, prompts in prompts_dict.items():
            show = prompts if final else prompts[:1]
            for prompt in show:
                formatted = f"User: {prompt}\n\nAssistant:"
                inputs = self.tokenizer(formatted, return_tensors="pt", truncation=True,
                                        max_length=self.max_new_tokens).to(device)
                try:
                    outputs = self.model.simple_generate(
                        input_ids=inputs["input_ids"], max_new_tokens=self.max_new_tokens,
                        temperature=0.7, top_k=50, top_p=0.9,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
                    response = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:],
                                                     skip_special_tokens=True)
                except Exception as e:
                    response = f"[Error: {str(e)[:80]}]"
                logger.info(f"  [{domain}] Q: {prompt[:80]}")
                logger.info(f"  [{domain}] A: {response[:250]}")
        self.model.train()
# ════════════════════════════════════════════════════════════════════════════
#  Eval Prompts
# ════════════════════════════════════════════════════════════════════════════
EVAL_PROMPTS = {
    "math": [
        "Martha has 18 crayons. She lost half of them, so she bought a new set of 20 crayons. How many crayons in total does Martha have after the purchase?",
        "The four-digit numeral 3AA1 is divisible by 9. What digit does A represent?",
        "Find the remainder when 2^100 is divided by 7.",
        "Solve: 3x + 7 = 22. What is x?",
    ],
    "coding": [
        "Given an array of integers, implement insertion sort in Python.",
        "Write a Python function to find the longest common subsequence of two strings.",
    ],
    "conversation": [
        "What are the key differences between renewable and non-renewable energy sources?",
        "What is the difference between machine learning and deep learning?",
    ],
    "reasoning": [
        "A bat and a ball cost $1.10 in total. The bat costs $1 more than the ball. How much does the ball cost?",
        "If P implies Q, and Q is false, what can we say about P?",
    ],
    "greetings": [
        "Hello! How are you today?",
        "Hi, can you help me with something?",
    ],
}
# ════════════════════════════════════════════════════════════════════════════
#  Dataset Loading  —  more data everywhere
# ════════════════════════════════════════════════════════════════════════════
def load_pretrain_dataset(cfg, tokenizer):
    logger.info("Loading pretraining corpus...")
    corpus_path = cfg.pretrain_corpus
    if IS_KAGGLE and not os.path.exists(corpus_path):
        corpus_path = "/kaggle/input/pretraining-corpus/pretraining_corpus.jsonl"
    if not os.path.exists(corpus_path):
        texts = ["Mathematics studies numbers and shapes."] * 1000
        ds = Dataset.from_dict({"text": texts})
    else:
        ds = load_dataset("json", data_files=corpus_path, split="train")
        if len(ds) > cfg.pretrain_max_samples:
            ds = ds.shuffle(seed=42).select(range(cfg.pretrain_max_samples))
    split = ds.train_test_split(test_size=cfg.pretrain_eval_split, seed=42)
    logger.info(f"Pretrain: {len(split['train']):,} train | {len(split['test']):,} eval")
    return split["train"], split["test"]
def load_sft_dataset(cfg):
    logger.info("Loading SFT dataset...")
    data_dir = cfg.sft_data_dir
    if IS_KAGGLE and not os.path.isdir(data_dir):
        for cand in ["/kaggle/input/datasets/abhishek0706/sft-dataset",
                     "/kaggle/input/datasets/abhishekgandhiau/sft-dataset-v1",
                     "/kaggle/input/datasets/abhishekgandhi0706/sft-dataset",
                     "/kaggle/input/sft-dataset/SFT_dataset", "/kaggle/input/sft-dataset"]:
            if os.path.isdir(cand):
                data_dir = cand
                break
    domain_files = {"math": "math_records.jsonl", "coding": "coding_records.jsonl",
                    "conversation": "conversation_records.jsonl", "reasoning": "reasoning_records.jsonl",
                    "greetings": "greetings_records.jsonl"}
    use_all = {"greetings"}
    all_records = []
    for domain, filename in domain_files.items():
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            continue
        records = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        total = len(records)
        if domain not in use_all and len(records) > cfg.sft_max_samples_per_domain:
            import random; random.seed(42); random.shuffle(records)
            records = records[:cfg.sft_max_samples_per_domain]
        logger.info(f"  {domain}: {len(records):,}/{total:,}")
        for r in records:
            r["domain"] = domain
        all_records.extend(records)
    if not all_records:
        all_records = [{"prompt": "What is 2+2?", "thinking": "4", "answer": "4", "domain": "math"}] * 100
    import random; random.seed(42); random.shuffle(all_records)
    logger.info(f"Total SFT: {len(all_records):,}")
    return Dataset.from_list(all_records)
def format_sft_text(example):
    p = example.get("prompt", "")
    t = example.get("thinking", "")
    a = example.get("answer", "")
    if len(t) > 3000:
        t = t[:1500] + " ... " + t[-1500:]
    if len(a) > 2000:
        a = a[:2000]
    return {"text": f"User: {p}\n\nAssistant: <think>{t}</think>\n<answer>{a}</answer>"}
def create_grpo_dataset(sft_dataset, cfg):
    def fmt(ex):
        return {"prompt": f"User: {ex.get('prompt','')}\n\nAssistant:", "solution": ex.get("answer", "")}
    ds = sft_dataset.map(fmt)
    if len(ds) > cfg.grpo_max_dataset_size:
        ds = ds.shuffle(seed=42).select(range(cfg.grpo_max_dataset_size))
    logger.info(f"GRPO dataset: {len(ds):,}")
    return ds
# ════════════════════════════════════════════════════════════════════════════
#  Reward Functions
# ════════════════════════════════════════════════════════════════════════════
def format_reward_func(completions, **kwargs):
    rewards = []
    for c in completions:
        text = " ".join(m.get("content", "") for m in c if isinstance(m, dict)) if isinstance(c, list) else str(c)
        r = 0.0
        has_think = bool(re.search(r"<think>.*?</think>", text, re.DOTALL))
        has_answer = bool(re.search(r"<answer>.*?</answer>", text, re.DOTALL))
        if has_think and has_answer:
            r += 1.0
            if text.find("<think>") < text.find("<answer>"):
                r += 0.5
        elif has_think or has_answer:
            r += 0.3
        rewards.append(r)
    return rewards
def length_reward_func(completions, **kwargs):
    rewards = []
    for c in completions:
        text = " ".join(m.get("content", "") for m in c if isinstance(m, dict)) if isinstance(c, list) else str(c)
        w = len(text.split())
        rewards.append(1.0 if 20 <= w <= 200 else 0.1 if w < 10 else 0.4 if w > 300 else 0.7)
    return rewards
def reasoning_quality_reward_func(completions, **kwargs):
    rewards = []
    for c in completions:
        text = " ".join(m.get("content", "") for m in c if isinstance(m, dict)) if isinstance(c, list) else str(c)
        r = 0.0
        m = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
        if m:
            reasoning = m.group(1).strip()
            w = len(reasoning.split())
            if w >= 10: r += 0.5
            if w >= 30: r += 0.3
            indicators = ["step", "first", "then", "therefore", "because", "since", "thus", "let me", "we can"]
            r += min(sum(1 for s in indicators if s in reasoning.lower()) * 0.1, 0.5)
        rewards.append(r)
    return rewards
def repetition_penalty_reward_func(completions, **kwargs):
    rewards = []
    for c in completions:
        text = " ".join(m.get("content", "") for m in c if isinstance(m, dict)) if isinstance(c, list) else str(c)
        if len(text.strip()) < 5:
            rewards.append(0.0); continue
        r = 1.0
        words = text.lower().split()
        if len(words) >= 4:
            fg = [tuple(words[i:i+4]) for i in range(len(words)-3)]
            rr = 1.0 - len(set(fg)) / len(fg)
            if rr > 0.6: r -= 0.5
            elif rr > 0.4: r -= 0.3
        rewards.append(max(r, 0.0))
    return rewards
def correctness_reward_func(completions, **kwargs):
    rewards = []
    solutions = kwargs.get("solution", [None] * len(completions))
    for i, c in enumerate(completions):
        text = " ".join(m.get("content", "") for m in c if isinstance(m, dict)) if isinstance(c, list) else str(c)
        gt = solutions[i] if i < len(solutions) and solutions[i] else None
        if not gt:
            rewards.append(0.0); continue
        am = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
        if not am:
            rewards.append(0.0); continue
        pred = am.group(1).strip()
        gt_t = str(gt).strip()
        try:
            if abs(float(pred.replace(",", "")) - float(gt_t.replace(",", ""))) < 1e-6:
                rewards.append(1.0); continue
        except Exception:
            pass
        pt = set(pred.lower().split()); gt_s = set(gt_t.lower().split())
        if pt and gt_s:
            o = pt & gt_s; p = len(o) / len(pt); r = len(o) / len(gt_s)
            rewards.append(min(2 * p * r / (p + r), 1.0) if (p + r) > 0 else 0.0)
        else:
            rewards.append(0.0)
    return rewards
# ════════════════════════════════════════════════════════════════════════════
#  Phase 0: Pretraining
# ════════════════════════════════════════════════════════════════════════════
def run_pretraining(model, tokenizer, cfg, device, state_mgr):
    logger.info("\n" + "=" * 70 + "\nPHASE 0: PRETRAINING\n" + "=" * 70)
    pretrain_train_ds, pretrain_eval_ds = load_pretrain_dataset(cfg, tokenizer)
    pretrain_path = os.path.join(cfg.output_dir, "pretrain")
    _ensure_clean_distributed_state()
    sft_sig = inspect.signature(SFTConfig.__init__)
    sft_params = set(sft_sig.parameters.keys())
    sft_args = dict(
        output_dir=pretrain_path,
        per_device_train_batch_size=cfg.pretrain_batch_size,
        per_device_eval_batch_size=cfg.pretrain_batch_size * 2,
        gradient_accumulation_steps=cfg.pretrain_grad_accum,
        learning_rate=cfg.pretrain_lr,
        max_steps=cfg.pretrain_max_steps,
        warmup_steps=cfg.pretrain_warmup_steps,
        weight_decay=cfg.pretrain_weight_decay,
        logging_steps=cfg.pretrain_logging_steps,
        save_steps=cfg.pretrain_save_steps,
        eval_strategy="steps",
        eval_steps=cfg.pretrain_eval_steps,
        save_total_limit=cfg.save_total_limit,
        bf16=False, fp16=True,                       # ← P100: FP16 only
        report_to="none",
        gradient_checkpointing=True,
        lr_scheduler_type="cosine",
        dataloader_num_workers=cfg.dataloader_num_workers,
        dataloader_pin_memory=True,
        load_best_model_at_end=False,
        disable_tqdm=True,
    )
    opt = {}
    if "max_seq_length" in sft_params: opt["max_seq_length"] = cfg.max_seq_len
    if "dataset_text_field" in sft_params: opt["dataset_text_field"] = "text"
    if "packing" in sft_params: opt["packing"] = True      # ← SPEED: packing ON
    sft_config = SFTConfig(**sft_args, **opt)
    model._set_gradient_checkpointing(True)
    best_cb = BestModelCallback(cfg.output_dir, "pretrain", tokenizer)
    val_cb = ValidationLoggerCallback("Pretrain")
    cbs = [PrintProgressCallback("Pretrain"),
           PipelineCallback(model, tokenizer, EVAL_PROMPTS, "Pretrain", cfg.pretrain_save_steps, 128),
           StorageMonitorCallback(cfg.output_dir), val_cb, best_cb]
    tk = dict(model=model, args=sft_config, train_dataset=pretrain_train_ds,
              eval_dataset=pretrain_eval_ds, callbacks=cbs)
    ti = set(inspect.signature(SFTTrainer.__init__).parameters.keys())
    tk["processing_class" if "processing_class" in ti else "tokenizer"] = tokenizer
    trainer = SFTTrainer(**tk)
    logger.info(f"Pretrain: single P100 | eff_batch={cfg.pretrain_batch_size * cfg.pretrain_grad_accum} | packing=ON")
    resume = find_latest_checkpoint(pretrain_path)
    trainer.train(resume_from_checkpoint=resume)
    enforce_storage_limit(cfg.output_dir, "save pretrain_model")
    save_path = os.path.join(cfg.output_dir, "pretrain_model")
    trainer.save_model(save_path); tokenizer.save_pretrained(save_path)
    cleanup_checkpoints(pretrain_path, keep_last=0)
    state_mgr.mark_complete("pretrain", best_eval_loss=val_cb.best_eval_loss,
                            steps=trainer.state.global_step)
    del trainer; torch.cuda.empty_cache(); gc.collect()
    return model
# ════════════════════════════════════════════════════════════════════════════
#  Phase 1: SFT
# ════════════════════════════════════════════════════════════════════════════
def run_sft(model, tokenizer, cfg, device, state_mgr):
    logger.info("\n" + "=" * 70 + "\nPHASE 1: SFT\n" + "=" * 70)
    raw_ds = load_sft_dataset(cfg)
    split = raw_ds.train_test_split(test_size=cfg.sft_eval_split, seed=42)
    train_ds = split["train"].map(format_sft_text, remove_columns=split["train"].column_names)
    eval_ds = split["test"].map(format_sft_text, remove_columns=split["test"].column_names)
    logger.info(f"SFT Train: {len(train_ds):,} | Eval: {len(eval_ds):,}")
    sft_path = os.path.join(cfg.output_dir, "sft")
    _ensure_clean_distributed_state()
    sft_sig = inspect.signature(SFTConfig.__init__)
    sft_params = set(sft_sig.parameters.keys())
    sft_args = dict(
        output_dir=sft_path,
        per_device_train_batch_size=cfg.sft_batch_size,
        per_device_eval_batch_size=cfg.sft_batch_size * 2,
        gradient_accumulation_steps=cfg.sft_grad_accum,
        learning_rate=cfg.sft_lr,
        max_steps=cfg.sft_max_steps,
        warmup_steps=cfg.sft_warmup_steps,
        weight_decay=cfg.sft_weight_decay,
        max_grad_norm=cfg.sft_max_grad_norm,
        logging_steps=cfg.sft_logging_steps,
        save_steps=cfg.sft_save_steps,
        eval_strategy="steps",
        eval_steps=cfg.sft_eval_steps,
        save_total_limit=cfg.save_total_limit,
        bf16=False, fp16=True,                       # ← P100: FP16 only
        report_to="none",
        gradient_checkpointing=True,
        lr_scheduler_type="cosine",
        load_best_model_at_end=False,
        dataloader_num_workers=cfg.dataloader_num_workers,
        dataloader_pin_memory=True,
        disable_tqdm=True,
    )
    opt = {}
    if "max_seq_length" in sft_params: opt["max_seq_length"] = cfg.max_seq_len
    if "dataset_text_field" in sft_params: opt["dataset_text_field"] = "text"
    if "packing" in sft_params: opt["packing"] = True      # ← SPEED: packing ON
    sft_config = SFTConfig(**sft_args, **opt)
    model._set_gradient_checkpointing(True)
    best_cb = BestModelCallback(cfg.output_dir, "sft", tokenizer)
    val_cb = ValidationLoggerCallback("SFT")
    cbs = [PrintProgressCallback("SFT"),
           PipelineCallback(model, tokenizer, EVAL_PROMPTS, "SFT", cfg.inference_every_steps, cfg.max_seq_len),
           StorageMonitorCallback(cfg.output_dir), val_cb, best_cb]
    tk = dict(model=model, args=sft_config, train_dataset=train_ds, eval_dataset=eval_ds, callbacks=cbs)
    ti = set(inspect.signature(SFTTrainer.__init__).parameters.keys())
    tk["processing_class" if "processing_class" in ti else "tokenizer"] = tokenizer
    trainer = SFTTrainer(**tk)
    logger.info(f"SFT: single P100 | eff_batch={cfg.sft_batch_size * cfg.sft_grad_accum} | "
                f"max_steps={cfg.sft_max_steps} | seq={cfg.max_seq_len} | packing=ON")
    resume = find_latest_checkpoint(sft_path)
    trainer.train(resume_from_checkpoint=resume)
    enforce_storage_limit(cfg.output_dir, "save sft_model")
    save_path = os.path.join(cfg.output_dir, "sft_model")
    trainer.save_model(save_path); tokenizer.save_pretrained(save_path)
    cleanup_checkpoints(sft_path, keep_last=0)
    pretrain_model_path = os.path.join(cfg.output_dir, "pretrain_model")
    if os.path.isdir(pretrain_model_path):
        shutil.rmtree(pretrain_model_path, ignore_errors=True)
    state_mgr.mark_complete("sft", best_eval_loss=val_cb.best_eval_loss,
                            steps=trainer.state.global_step)
    check_output_storage(cfg.output_dir)
    del trainer; torch.cuda.empty_cache(); gc.collect()
    return model, raw_ds


# ════════════════════════════════════════════════════════════════════════════
#  Helper: monkey-patch TRL's create_model_from_path to prevent auto ref
#  model creation for custom model types (would fail with empty _name_or_path)
# ════════════════════════════════════════════════════════════════════════════
def _patch_trl_no_ref_model():
    """
    Returns a context-manager-like pair (patch, unpatch) that replaces
    trl.trainer.grpo_trainer.create_model_from_path with a no-op so that
    GRPOTrainer.__init__ skips automatic ref-model creation.
    """
    import trl.trainer.grpo_trainer as _grpo_mod
    _orig = getattr(_grpo_mod, "create_model_from_path", None)
    def _noop(*_args, **_kwargs):
        return None
    def patch():
        if _orig is not None:
            _grpo_mod.create_model_from_path = _noop
    def unpatch():
        if _orig is not None:
            _grpo_mod.create_model_from_path = _orig
    return patch, unpatch


def _nuclear_gpu_cleanup(model, device):
    """Move model to CPU, purge ALL GPU state, move model back."""
    model.cpu()
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    gc.collect()
    if torch.cuda.is_available():
        free_mem = torch.cuda.mem_get_info(0)[0] / 1e9
        total_mem = torch.cuda.mem_get_info(0)[1] / 1e9
        logger.info(f"  After cleanup: {free_mem:.1f} / {total_mem:.1f} GB free")
    model.to(device)
    model._set_gradient_checkpointing(True)
    torch.cuda.empty_cache()
    _ensure_clean_distributed_state()


# ════════════════════════════════════════════════════════════════════════════
#  Phase 2: GRPO  — single GPU, robust OOM fallback
# ════════════════════════════════════════════════════════════════════════════
def run_grpo(model, tokenizer, raw_sft_ds, cfg, device, state_mgr,
             ref_model_path: str = None):
    logger.info("\n" + "=" * 70 + "\nPHASE 2: GRPO\n" + "=" * 70)
    grpo_ds = create_grpo_dataset(raw_sft_ds, cfg)
    tokenizer.padding_side = "left"
    grpo_path = os.path.join(cfg.output_dir, "grpo")
    _ensure_clean_distributed_state()
    grpo_sig = inspect.signature(GRPOConfig.__init__)
    grpo_params = set(grpo_sig.parameters.keys())
    base_args = dict(
        output_dir=grpo_path,
        per_device_train_batch_size=cfg.grpo_batch_size,
        gradient_accumulation_steps=cfg.grpo_grad_accum,
        learning_rate=cfg.grpo_lr,
        max_steps=cfg.grpo_max_steps,
        logging_steps=cfg.grpo_logging_steps,
        save_steps=cfg.grpo_save_steps,
        warmup_steps=cfg.grpo_warmup_steps,
        weight_decay=cfg.grpo_weight_decay,
        max_grad_norm=cfg.grpo_max_grad_norm,
        bf16=False, fp16=True,                       # ← P100: FP16 only
        report_to="none",
        save_total_limit=cfg.save_total_limit,
        gradient_checkpointing=True,
        lr_scheduler_type="cosine",
        dataloader_pin_memory=True,
        disable_tqdm=True,
    )
    opt = {}
    if "num_generations"       in grpo_params: opt["num_generations"]       = cfg.grpo_num_generations
    if "max_completion_length" in grpo_params: opt["max_completion_length"] = cfg.grpo_max_completion_length
    if "max_prompt_length"     in grpo_params: opt["max_prompt_length"]     = cfg.grpo_max_prompt_length
    if "beta"                  in grpo_params: opt["beta"]                  = cfg.grpo_beta
    if "remove_unused_columns" in grpo_params: opt["remove_unused_columns"] = False
    grpo_config = GRPOConfig(**base_args, **opt)
    model._set_gradient_checkpointing(True)

    # ── Validate ref_model_path ─────────────────────────────────────────────
    if ref_model_path is None:
        ref_model_path = os.path.join(cfg.output_dir, "sft_model")
    if not os.path.isdir(ref_model_path):
        if os.path.isfile(ref_model_path):
            ref_model_path = os.path.dirname(ref_model_path)
        else:
            raise FileNotFoundError(f"Reference model path does not exist: {ref_model_path}")
    logger.info(f"Reference model weights dir: {ref_model_path}")

    # ── FIX: Set _name_or_path so TRL's get_config_model_id() returns a
    #    valid path.  Without this, custom model types get an empty string
    #    which makes HuggingFace Hub validation fail. ────────────────────────
    model.config._name_or_path = ref_model_path

    # ── Reference model — check TRL API, load only if accepted ──────────
    gi = set(inspect.signature(GRPOTrainer.__init__).parameters.keys())
    ref_model_accepted = "ref_model" in gi
    ref_model = None

    if ref_model_accepted:
        # Older TRL: we can pass ref_model directly → load it
        ref_device = torch.device(device)
        ref_model = load_checkpoint_robust(model.config, ref_model_path, device=ref_device)
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad = False
        torch.cuda.empty_cache()
        logger.info(f"Reference model loaded on {ref_device} (beta={cfg.grpo_beta})")
    else:
        # Newer TRL: auto-creates ref_model internally.
        # Since _name_or_path is now set, TRL can auto-load via
        # create_model_from_path.  But to save VRAM we set beta=0
        # and monkey-patch to skip the ref model entirely.
        grpo_config.beta = 0.0
        logger.info("  ref_model not accepted as param — set beta=0.0 "
                     "(reward-only GRPO, no KL penalty)")

    best_cb = BestModelCallback(cfg.output_dir, "grpo", tokenizer)
    cbs = [PrintProgressCallback("GRPO"),
           PipelineCallback(model, tokenizer, EVAL_PROMPTS, "GRPO",
                            cfg.inference_every_steps, cfg.grpo_max_completion_length),
           StorageMonitorCallback(cfg.output_dir, 200), best_cb]
    reward_funcs = [format_reward_func, length_reward_func, reasoning_quality_reward_func,
                    repetition_penalty_reward_func, correctness_reward_func]

    gk = dict(model=model, args=grpo_config, train_dataset=grpo_ds,
              reward_funcs=reward_funcs, callbacks=cbs)
    if ref_model_accepted and ref_model is not None:
        gk["ref_model"] = ref_model
    if "processing_class" in gi:
        gk["processing_class"] = tokenizer
    elif "tokenizer" in gi:
        gk["tokenizer"] = tokenizer
    else:
        gk["processing_class"] = tokenizer

    # ── Build GRPOTrainer — for newer TRL that auto-creates ref model,
    #    we need the monkey-patch when ref_model param is NOT accepted ───────
    patch_fn, unpatch_fn = _patch_trl_no_ref_model()
    if not ref_model_accepted:
        patch_fn()
    try:
        grpo_trainer = GRPOTrainer(**gk)
    finally:
        unpatch_fn()

    # Ensure ref_model is None if we patched (belt and suspenders)
    if not ref_model_accepted:
        if hasattr(grpo_trainer, 'ref_model') and grpo_trainer.ref_model is not None:
            grpo_trainer.ref_model.cpu()
            del grpo_trainer.ref_model
            grpo_trainer.ref_model = None
            torch.cuda.empty_cache()

    logger.info(f"GRPO: single P100 | eff_batch={cfg.grpo_batch_size * cfg.grpo_grad_accum} | "
                f"prompt={cfg.grpo_max_prompt_length} | completion={cfg.grpo_max_completion_length}")
    resume = find_latest_checkpoint(grpo_path)
    grpo_succeeded = False

    # ── Attempt 1: normal settings ──
    try:
        grpo_trainer.train(resume_from_checkpoint=resume)
        grpo_succeeded = True
    except torch.cuda.OutOfMemoryError:
        logger.warning("OOM during GRPO attempt 1! Performing full GPU memory reset...")

        # --- Free trainer and ref model completely ---
        try:
            if hasattr(grpo_trainer, 'ref_model') and grpo_trainer.ref_model is not None:
                grpo_trainer.ref_model.cpu()
            del grpo_trainer
        except Exception:
            pass
        if ref_model is not None:
            ref_model.cpu()
            del ref_model
            ref_model = None
        gk.pop("ref_model", None)

        # --- Nuclear GPU cleanup ---
        _nuclear_gpu_cleanup(model, device)

        # ── Attempt 2: conservative (prompt=96, completion=128, total=224) ──
        logger.info("  Fallback attempt 2: bs=1, prompt=96, completion=128, beta=0.0")
        fallback_args = {**base_args}
        fallback_args["per_device_train_batch_size"] = 1
        fallback_args["gradient_accumulation_steps"] = 16
        fallback_opt = {}
        if "num_generations"       in grpo_params: fallback_opt["num_generations"]       = 2
        if "max_completion_length" in grpo_params: fallback_opt["max_completion_length"] = 128
        if "max_prompt_length"     in grpo_params: fallback_opt["max_prompt_length"]     = 96
        if "beta"                  in grpo_params: fallback_opt["beta"]                  = 0.0
        if "remove_unused_columns" in grpo_params: fallback_opt["remove_unused_columns"] = False
        grpo_config2 = GRPOConfig(**fallback_args, **fallback_opt)
        # Force beta=0 even if param name changed
        grpo_config2.beta = 0.0

        gk["model"] = model
        gk["args"] = grpo_config2
        gk["reward_funcs"] = [format_reward_func, length_reward_func, correctness_reward_func]
        gk["callbacks"] = [PrintProgressCallback("GRPO-fallback2"),
                           StorageMonitorCallback(cfg.output_dir, 200), best_cb]

        try:
            # Monkey-patch to prevent TRL from auto-creating ref model
            patch_fn()
            try:
                grpo_trainer = GRPOTrainer(**gk)
            finally:
                unpatch_fn()
            # Ensure no ref model lingering
            if hasattr(grpo_trainer, 'ref_model') and grpo_trainer.ref_model is not None:
                grpo_trainer.ref_model.cpu()
                del grpo_trainer.ref_model
                grpo_trainer.ref_model = None
                torch.cuda.empty_cache()
            grpo_trainer.args.beta = 0.0
            grpo_trainer.train()
            grpo_succeeded = True

        except torch.cuda.OutOfMemoryError:
            logger.error("OOM on attempt 2! Trying minimal config (prompt=64, completion=64)...")
            try:
                if hasattr(grpo_trainer, 'ref_model') and grpo_trainer.ref_model is not None:
                    grpo_trainer.ref_model.cpu()
                del grpo_trainer
            except Exception:
                pass

            _nuclear_gpu_cleanup(model, device)

            # ── Attempt 3: absolute minimum ──
            fallback_args3 = {**base_args}
            fallback_args3["per_device_train_batch_size"] = 1
            fallback_args3["gradient_accumulation_steps"] = 16
            fallback_args3["max_steps"] = 500
            fallback_opt3 = {}
            if "num_generations"       in grpo_params: fallback_opt3["num_generations"]       = 2
            if "max_completion_length" in grpo_params: fallback_opt3["max_completion_length"] = 64
            if "max_prompt_length"     in grpo_params: fallback_opt3["max_prompt_length"]     = 64
            if "beta"                  in grpo_params: fallback_opt3["beta"]                  = 0.0
            if "remove_unused_columns" in grpo_params: fallback_opt3["remove_unused_columns"] = False
            grpo_config3 = GRPOConfig(**fallback_args3, **fallback_opt3)
            grpo_config3.beta = 0.0

            gk["model"] = model
            gk["args"] = grpo_config3
            gk["callbacks"] = [PrintProgressCallback("GRPO-fallback3"),
                               StorageMonitorCallback(cfg.output_dir, 200), best_cb]

            try:
                patch_fn()
                try:
                    grpo_trainer = GRPOTrainer(**gk)
                finally:
                    unpatch_fn()
                if hasattr(grpo_trainer, 'ref_model') and grpo_trainer.ref_model is not None:
                    grpo_trainer.ref_model.cpu()
                    del grpo_trainer.ref_model
                    grpo_trainer.ref_model = None
                    torch.cuda.empty_cache()
                grpo_trainer.args.beta = 0.0
                grpo_trainer.train()
                grpo_succeeded = True
            except torch.cuda.OutOfMemoryError:
                logger.error("OOM on attempt 3! Skipping GRPO — saving current model as final.")
                grpo_succeeded = False

    # ── Save final model ──
    final_path = os.path.join(cfg.output_dir, "final_model")
    os.makedirs(final_path, exist_ok=True)
    enforce_storage_limit(cfg.output_dir, "save final_model")
    if grpo_succeeded:
        grpo_trainer.save_model(final_path)
        tokenizer.save_pretrained(final_path)
        state_mgr.mark_complete("grpo", steps=grpo_trainer.state.global_step)
        try:
            del grpo_trainer
        except Exception:
            pass
    else:
        # Save the SFT model as "final" since GRPO couldn't run
        save_model = model.module if hasattr(model, "module") else model
        save_model.save_pretrained(final_path)
        tokenizer.save_pretrained(final_path)
        state_mgr.mark_complete("grpo", steps=0)
    cleanup_checkpoints(grpo_path, keep_last=0)
    local_sft_model = os.path.join(cfg.output_dir, "sft_model")
    if os.path.isdir(local_sft_model):
        shutil.rmtree(local_sft_model, ignore_errors=True)
    check_output_storage(cfg.output_dir)
    torch.cuda.empty_cache(); gc.collect()
    return model
# ════════════════════════════════════════════════════════════════════════════
#  Final Evaluation
# ════════════════════════════════════════════════════════════════════════════
def run_final_eval(model, tokenizer, cfg, device):
    logger.info("\n" + "=" * 70 + "\nFINAL EVALUATION\n" + "=" * 70)
    model.eval()
    test_prompts = {
        "math": ["A store sells notebooks for $3 each. Buy 5+ get 20% off. How much do 7 cost?",
                  "What is the sum of all integers from 1 to 100?"],
        "coding": ["Write a Python function to compute factorial using recursion."],
        "conversation": ["Explain the greenhouse effect and its role in climate change."],
        "reasoning": ["If it rains, the ground gets wet. The ground is wet. Did it necessarily rain?"],
        "greetings": ["Good morning! What can you do?"],
    }
    for domain, prompts in test_prompts.items():
        logger.info(f"\n--- [{domain.upper()}] ---")
        for prompt in prompts:
            formatted = f"User: {prompt}\n\nAssistant:"
            inputs = tokenizer(formatted, return_tensors="pt", truncation=True,
                               max_length=cfg.max_seq_len).to(device)
            with torch.no_grad():
                outputs = model.simple_generate(
                    inputs["input_ids"], max_new_tokens=cfg.max_seq_len,
                    temperature=0.7, top_p=0.9, eos_token_id=tokenizer.eos_token_id)
            response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            logger.info(f"  Q: {prompt}")
            logger.info(f"  A: {response}")
# ════════════════════════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════════════════════════
def main():
    train_start = time.time()
    logger.info("\n" + "=" * 70)
    logger.info("HybridMoRMoE Full Pipeline — P100 x1 (SINGLE GPU, 16 GB)")
    logger.info("=" * 70)
    has_gpu, num_gpus = setup_gpu()
    device = "cuda" if has_gpu else "cpu"
    cfg = PipelineConfig()
    cfg.model_size    = os.environ.get("MODEL_SIZE",    "medium")
    cfg.skip_pretrain = os.environ.get("SKIP_PRETRAIN", "1") == "1"
    cfg.skip_sft      = os.environ.get("SKIP_SFT",      "1") == "1"
    cfg.num_gpus      = 1                                # always 1 for P100
    if IS_KAGGLE:
        cfg.output_dir      = "/kaggle/working/hybrid_mor_moe_P100"
        cfg.sft_data_dir    = "/kaggle/input/datasets/abhishekgandhiau/sft-dataset-v1"
        cfg.pretrain_corpus = "/kaggle/input/pretraining-corpus/pretraining_corpus.jsonl"
        cfg.tokenizer_path  = "/kaggle/input/qwen-tokenizer/Qwen2.5-0.5B-Instruct"
    cfg = adjust_config_for_model_size(cfg)
    os.makedirs(cfg.output_dir, exist_ok=True)
    state_mgr = PipelineStateManager(cfg.output_dir)
    logger.info(f"  [Checkpoint] {state_mgr.summary()}")
    if state_mgr.is_complete("pretrain"):
        cfg.skip_pretrain = True
        logger.info("  [Checkpoint] pretrain already done → skip")
    if state_mgr.is_complete("sft"):
        cfg.skip_sft = True
        logger.info("  [Checkpoint] sft already done → skip")
    logger.info(f"Model: {cfg.model_size} | GPU: P100 x1 | Seq: {cfg.max_seq_len} | SFT steps: {cfg.sft_max_steps}")
    logger.info(f"Data: pretrain={cfg.pretrain_max_samples//1000}K  sft/dom={cfg.sft_max_samples_per_domain//1000}K  "
                f"grpo={cfg.grpo_max_dataset_size//1000}K")
    logger.info(f"Skip pretrain: {cfg.skip_pretrain} | Skip SFT: {cfg.skip_sft}")
    check_output_storage(cfg.output_dir)
    # ── Tokenizer ──
    if os.path.isdir(cfg.tokenizer_path):
        tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_path, trust_remote_code=True, local_files_only=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_hf_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model_config = HybridMoRMoEConfig(
        model_size=cfg.model_size, max_seq_len=cfg.max_seq_len, dropout=cfg.dropout)
    model_config.vocab_size = len(tokenizer)
    # ── Pre-trained SFT model path (Kaggle input) ──
    INPUT_SFT_MODEL_DIR = "/kaggle/input/models/abhishekgandhiau/hybrid-mor-moe/transformers/default/1"
    pretrain_model_path = os.path.join(cfg.output_dir, "pretrain_model")
    raw_sft_ds = None
    ref_model_path_for_grpo = None
    # ── Model loading ──
    if cfg.skip_sft and os.path.isdir(INPUT_SFT_MODEL_DIR):
        logger.info(f"Loading existing SFT model: {INPUT_SFT_MODEL_DIR}")
        model = load_checkpoint_robust(model_config, INPUT_SFT_MODEL_DIR, device=device)
        raw_sft_ds = load_sft_dataset(cfg)
        ref_model_path_for_grpo = INPUT_SFT_MODEL_DIR
    elif cfg.skip_pretrain and os.path.isdir(pretrain_model_path):
        logger.info(f"Loading existing pretrain model: {pretrain_model_path}")
        model = load_checkpoint_robust(model_config, pretrain_model_path, device=device)
    else:
        model = HybridMoRMoEForCausalLM(model_config)
        model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: {total_params:,} params ({total_params/1e6:.1f}M)")
    with torch.no_grad():
        test_ids = torch.randint(0, model_config.vocab_size, (2, 32), device=device)
        test_out = model(test_ids, labels=test_ids, return_dict=True)
        logger.info(f"Forward pass OK, loss={test_out.loss.item():.4f}")
    del test_ids, test_out; torch.cuda.empty_cache()
    # ── Phase 0: Pretrain ──
    if not cfg.skip_pretrain:
        model = run_pretraining(model, tokenizer, cfg, device, state_mgr)
    else:
        logger.info("\nPHASE 0: PRETRAINING — SKIPPED")
    ckpt_dir = os.path.join(cfg.output_dir, "pretrain")
    if os.path.isdir(ckpt_dir):
        cleanup_checkpoints(ckpt_dir, keep_last=0)
    # ── Phase 1: SFT ──
    if not cfg.skip_sft:
        model, raw_sft_ds = run_sft(model, tokenizer, cfg, device, state_mgr)
        ref_model_path_for_grpo = os.path.join(cfg.output_dir, "sft_model")
    else:
        logger.info("\nPHASE 1: SFT — SKIPPED")
        if raw_sft_ds is None:
            raw_sft_ds = load_sft_dataset(cfg)
    # ── Phase 2: GRPO ──
    model = run_grpo(model, tokenizer, raw_sft_ds, cfg, device, state_mgr,
                     ref_model_path=ref_model_path_for_grpo)
    run_final_eval(model, tokenizer, cfg, device)
    check_output_storage(cfg.output_dir)
    total_time = time.time() - train_start
    logger.info("\n" + "=" * 70)
    logger.info("PIPELINE COMPLETE!")
    logger.info(f"  Model: {cfg.model_size} ({total_params/1e6:.1f}M) | GPU: P100 x1")
    logger.info(f"  Wall time: {total_time/3600:.2f}h")
    logger.info(f"  {state_mgr.summary()}")
    logger.info(f"  Final model: {os.path.join(cfg.output_dir, 'final_model')}")
    logger.info("=" * 70)
    return model, tokenizer
if __name__ == "__main__":
    main()
