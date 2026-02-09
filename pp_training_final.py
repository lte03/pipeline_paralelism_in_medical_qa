import os
import json
import time
import csv
import torch
import torch.nn as nn
import deepspeed
from datasets import load_from_disk
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from deepspeed.pipe import PipelineModule
import random
import numpy as np
import psutil
from tqdm import tqdm

try:
    import pynvml
    pynvml.nvmlInit()
    NVML_AVAILABLE = True
except:
    NVML_AVAILABLE = False

# ==================== GLOBAL AYARLAR ====================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ==================== AYARLAR ====================
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

EPOCHS = 2
MICRO_BSZ = 2          # Multi-GPU micro batch size
GRAD_ACCUM = 16         # Gradient accumulation steps
SINGLE_GPU_BSZ = 1     # Tek GPU için micro batch size (grad accum ile effective=8)
SINGLE_GPU_GRAD_ACCUM = 8  # Tek GPU gradient accumulation
LR = 2e-5
ZERO_STAGE = 1
SCHEDULE = "1f1b"

TOKENIZED_TRAIN_PATH = "./tokenized_data/train"
SAVE_DIR = "./pp_ckpt"
METRICS_DIR = "./training_metrics"

# ==================== METRICS COLLECTION ====================
class MetricsCollector:
    def __init__(self, num_gpus, save_dir=METRICS_DIR):
        self.num_gpus = num_gpus
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        self.start_time = None
        self.metrics_log = []
        
        # CSV dosyası hazırla
        self.csv_path = os.path.join(save_dir, "training_metrics.csv")
        self.json_path = os.path.join(save_dir, "training_summary.json")
        
    def start(self):
        self.start_time = time.time()
        
    def get_gpu_metrics(self):
        """Tüm GPU'ların kullanım ve bellek bilgisini al"""
        gpu_metrics = []
        
        if NVML_AVAILABLE:
            for i in range(self.num_gpus):
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    gpu_metrics.append({
                        "gpu_id": i,
                        "gpu_util_percent": util.gpu,
                        "mem_used_gb": mem.used / (1024**3),
                        "mem_total_gb": mem.total / (1024**3),
                        "mem_util_percent": (mem.used / mem.total) * 100
                    })
                except:
                    gpu_metrics.append({"gpu_id": i, "error": "unavailable"})
        else:
            # Fallback: nvidia-smi subprocess kullan (tüm GPU'ları görebilir)
            try:
                import subprocess
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used,memory.total', '--format=csv,noheader,nounits'],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    for line in result.stdout.strip().split('\n'):
                        parts = [p.strip() for p in line.split(',')]
                        if len(parts) >= 4:
                            gpu_id = int(parts[0])
                            if gpu_id < self.num_gpus:
                                gpu_util = float(parts[1]) if parts[1] != '[N/A]' else -1
                                mem_used = float(parts[2]) / 1024  # MiB to GB
                                mem_total = float(parts[3]) / 1024  # MiB to GB
                                gpu_metrics.append({
                                    "gpu_id": gpu_id,
                                    "gpu_util_percent": gpu_util,
                                    "mem_used_gb": mem_used,
                                    "mem_total_gb": mem_total,
                                    "mem_util_percent": (mem_used / mem_total) * 100 if mem_total > 0 else 0
                                })
                else:
                    raise Exception("nvidia-smi failed")
            except Exception as e:
                # Son fallback: torch.cuda kullan
                for i in range(self.num_gpus):
                    if torch.cuda.is_available():
                        try:
                            mem_used = torch.cuda.memory_allocated(i) / (1024**3)
                            mem_total = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                            gpu_metrics.append({
                                "gpu_id": i,
                                "gpu_util_percent": -1,
                                "mem_used_gb": mem_used,
                                "mem_total_gb": mem_total,
                                "mem_util_percent": (mem_used / mem_total) * 100 if mem_total > 0 else 0
                            })
                        except:
                            gpu_metrics.append({"gpu_id": i, "error": "unavailable"})
        
        return gpu_metrics
    
    def get_ram_metrics(self):
        """RAM kullanım bilgisini al"""
        ram = psutil.virtual_memory()
        return {
            "ram_used_gb": ram.used / (1024**3),
            "ram_total_gb": ram.total / (1024**3),
            "ram_util_percent": ram.percent
        }
    
    def log_step(self, epoch, step, loss):
        """Her step için metrics kaydet"""
        elapsed = time.time() - self.start_time if self.start_time else 0
        
        gpu_metrics = self.get_gpu_metrics()
        ram_metrics = self.get_ram_metrics()
        
        # Flatten GPU metrics
        record = {
            "epoch": epoch,
            "step": step,
            "loss": loss,
            "elapsed_seconds": elapsed,
            "ram_used_gb": ram_metrics["ram_used_gb"],
            "ram_util_percent": ram_metrics["ram_util_percent"]
        }
        
        for gm in gpu_metrics:
            gpu_id = gm.get("gpu_id", 0)
            record[f"gpu{gpu_id}_util_percent"] = gm.get("gpu_util_percent", -1)
            record[f"gpu{gpu_id}_mem_used_gb"] = gm.get("mem_used_gb", -1)
            record[f"gpu{gpu_id}_mem_util_percent"] = gm.get("mem_util_percent", -1)
        
        self.metrics_log.append(record)
        
    def save_csv(self):
        """Tüm metrics'i CSV'ye kaydet"""
        if not self.metrics_log:
            return
            
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.metrics_log[0].keys())
            writer.writeheader()
            writer.writerows(self.metrics_log)
        
        print(f"📊 Metrics CSV kaydedildi: {self.csv_path}")
    
    def save_summary_json(self):
        """Ortalama istatistikleri JSON olarak kaydet"""
        if not self.metrics_log:
            return
        
        total_time = time.time() - self.start_time if self.start_time else 0
        
        # Ortalama hesapla
        avg_loss = sum(r["loss"] for r in self.metrics_log) / len(self.metrics_log)
        avg_ram_util = sum(r["ram_util_percent"] for r in self.metrics_log) / len(self.metrics_log)
        
        # GPU ortalamaları
        gpu_avgs = {}
        for i in range(self.num_gpus):
            key_util = f"gpu{i}_util_percent"
            key_mem = f"gpu{i}_mem_util_percent"
            
            util_vals = [r.get(key_util, 0) for r in self.metrics_log if r.get(key_util, -1) >= 0]
            mem_vals = [r.get(key_mem, 0) for r in self.metrics_log if r.get(key_mem, -1) >= 0]
            
            if util_vals:
                gpu_avgs[f"gpu{i}_avg_util_percent"] = sum(util_vals) / len(util_vals)
            if mem_vals:
                gpu_avgs[f"gpu{i}_avg_mem_util_percent"] = sum(mem_vals) / len(mem_vals)
        
        summary = {
            "total_training_time_seconds": total_time,
            "total_training_time_minutes": total_time / 60,
            "total_steps": len(self.metrics_log),
            "avg_loss": avg_loss,
            "avg_ram_util_percent": avg_ram_util,
            "num_gpus": self.num_gpus,
            **gpu_avgs
        }
        
        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"📊 Summary JSON kaydedildi: {self.json_path}")
        print(f"   └─ Toplam süre: {total_time/60:.2f} dakika")
        print(f"   └─ Ortalama loss: {avg_loss:.4f}")
        
        return summary

# ==================== MODEL PARSER (AUTO) ====================
def _get_attr_chain(obj, chain: str):
    cur = obj
    for part in chain.split("."):
        if not hasattr(cur, part):
            return None
        cur = getattr(cur, part)
    return cur

def find_first_attr_chain(obj, chains):
    for c in chains:
        got = _get_attr_chain(obj, c)
        if got is not None:
            return got, c
    return None, None

def parse_decoder_only_parts(causal_lm_model):
    embed, embed_path = find_first_attr_chain(causal_lm_model, [
        "model.embed_tokens",
        "model.model.embed_tokens",
        "transformer.wte",
        "gpt_neox.embed_in",
    ])
    if embed is None:
        raise RuntimeError("embed_tokens bulunamadı")

    blocks, blocks_path = find_first_attr_chain(causal_lm_model, [
        "model.layers",
        "model.model.layers",
        "transformer.h",
        "gpt_neox.layers",
    ])
    if blocks is None:
        raise RuntimeError("decoder layers bulunamadı")

    final_norm, norm_path = find_first_attr_chain(causal_lm_model, [
        "model.norm",
        "model.model.norm",
        "transformer.ln_f",
        "gpt_neox.final_layer_norm",
    ])

    lm_head, head_path = find_first_attr_chain(causal_lm_model, [
        "lm_head",
        "model.lm_head",
        "transformer.lm_head",
        "embed_out",
    ])
    if lm_head is None:
        raise RuntimeError("lm_head bulunamadı")

    rotary_emb, rot_path = find_first_attr_chain(causal_lm_model, [
        "model.rotary_emb",
        "model.model.rotary_emb",
        "transformer.rotary_emb",
    ])

    blocks = list(blocks)

    print(f"[Parser] embed: {embed_path}")
    print(f"[Parser] blocks: {blocks_path} (n={len(blocks)})")
    print(f"[Parser] norm: {norm_path}")
    print(f"[Parser] lm_head: {head_path}")
    if rotary_emb is not None:
        print(f"[Parser] rotary_emb bulundu: {rot_path} (NOT: dışarıdan kullanılmayacak)")

    return embed, blocks, final_norm, lm_head, rotary_emb

# ==================== PIPELINE STAGES (Multi-GPU için) ====================

class Stage0(nn.Module):
    def __init__(self, embed):
        super().__init__()
        self.embed = embed

    def forward(self, batch_tensor):
        if isinstance(batch_tensor, (tuple, list)):
            batch_tensor = batch_tensor[0]
        if batch_tensor.dim() == 1:
            batch_tensor = batch_tensor.unsqueeze(0)

        batch_size = batch_tensor.size(0)
        total_len = batch_tensor.size(1)
        seq_len = total_len // 3

        input_ids = batch_tensor[:, :seq_len].long()
        attention_mask = batch_tensor[:, seq_len:2*seq_len].long()
        labels = batch_tensor[:, 2*seq_len:].long()

        hidden_states = self.embed(input_ids)

        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)

        return (
            hidden_states.contiguous(),
            attention_mask.contiguous(),
            position_ids.contiguous(),
            labels.contiguous()
        )

class StageBlock(nn.Module):
    def __init__(self, block, rotary_emb):
        super().__init__()
        self.block = block
        self.rotary_emb = rotary_emb

    def _create_causal_mask(self, seq_len, dtype, device):
        mask = torch.full((seq_len, seq_len), torch.finfo(dtype).min, device=device)
        mask_cond = torch.arange(mask.size(-1), device=device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        return mask.to(dtype)

    def forward(self, args):
        hidden_states, attention_mask, position_ids, labels = args

        batch_size, seq_len, _ = hidden_states.shape
        dtype = hidden_states.dtype
        device = hidden_states.device

        padding_mask = attention_mask.unsqueeze(1).unsqueeze(1).to(dtype)
        padding_mask = (1.0 - padding_mask) * torch.finfo(dtype).min

        causal_mask = self._create_causal_mask(seq_len, dtype, device).unsqueeze(0).unsqueeze(0)

        combined_mask = padding_mask + causal_mask

        if self.rotary_emb is not None:
            cos, sin = self.rotary_emb(hidden_states, position_ids)
            position_embeddings = (cos, sin)
        else:
            position_embeddings = None

        out = self.block(
            hidden_states,
            attention_mask=combined_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            use_cache=False
        )
        hidden_states = out[0] if isinstance(out, (tuple, list)) else out

        return (
            hidden_states.contiguous(),
            attention_mask.contiguous(),
            position_ids.contiguous(),
            labels.contiguous()
        )

class StageLast(nn.Module):
    def __init__(self, final_norm, lm_head):
        super().__init__()
        self.final_norm = final_norm
        self.lm_head = lm_head
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, args):
        hidden_states, attention_mask, position_ids, labels = args

        if self.final_norm is not None:
            hidden_states = self.final_norm(hidden_states)

        logits = self.lm_head(hidden_states)

        # --- SHIFTING LOGIC ---
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss = self.loss_fn(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )
        return loss

# ==================== DEEPSPEED CONFIG ====================
def write_ds_config(use_pipeline=True):
    if use_pipeline:
        # Multi-GPU Pipeline mode
        cfg = {
            "train_micro_batch_size_per_gpu": MICRO_BSZ,
            "gradient_accumulation_steps": GRAD_ACCUM,
            "bf16": {"enabled": True},
            "fp16": {"enabled": False},
            "zero_optimization": {"stage": ZERO_STAGE},
            "optimizer": {"type": "AdamW", "params": {"lr": LR}},
            "gradient_clipping": 1.0,
            "pipeline": {"schedule": SCHEDULE}
        }
    else:
        # Single GPU mode
        cfg = {
            "train_micro_batch_size_per_gpu": SINGLE_GPU_BSZ,
            "gradient_accumulation_steps": SINGLE_GPU_GRAD_ACCUM,
            "bf16": {"enabled": True},
            "fp16": {"enabled": False},
            "zero_optimization": {"stage": ZERO_STAGE},
            "optimizer": {"type": "AdamW", "params": {"lr": LR}},
            "gradient_clipping": 1.0
        }
    
    with open("ds_config.json", "w") as f:
        json.dump(cfg, f, indent=2)

def create_pipeline_stages(embed, blocks, final_norm, lm_head, rotary_emb, num_gpus):
    n_blocks = len(blocks)

    layers = []
    layers.append(Stage0(embed))

    if num_gpus == 2:
        for b in blocks:
            layers.append(StageBlock(b, rotary_emb))
    else:
        middle_gpus = num_gpus - 2
        blocks_per_gpu = n_blocks // middle_gpus
        remaining = n_blocks % middle_gpus

        start_idx = 0
        for gpu_idx in range(middle_gpus):
            extra = 1 if gpu_idx < remaining else 0
            end_idx = start_idx + blocks_per_gpu + extra

            for i in range(start_idx, end_idx):
                layers.append(StageBlock(blocks[i], rotary_emb))

            start_idx = end_idx

    layers.append(StageLast(final_norm, lm_head))

    return layers

# ==================== COLLATE FUNCTIONS ====================
def collate_fn_pipeline(examples):
    """Pipeline parallelism için collate function"""
    input_ids = torch.tensor([e["input_ids"] for e in examples], dtype=torch.long)
    attention_mask = torch.tensor([e["attention_mask"] for e in examples], dtype=torch.long)
    labels = torch.tensor([e["labels"] for e in examples], dtype=torch.long)
    
    combined = torch.cat([input_ids, attention_mask, labels], dim=1)
    return (combined, torch.tensor([0]))

def collate_fn_single(examples):
    """Tek GPU için collate function"""
    input_ids = torch.tensor([e["input_ids"] for e in examples], dtype=torch.long)
    attention_mask = torch.tensor([e["attention_mask"] for e in examples], dtype=torch.long)
    labels = torch.tensor([e["labels"] for e in examples], dtype=torch.long)
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

# ==================== TRAINING FUNCTIONS ====================
def train_single_gpu(peft_model, train_ds, rank, world):
    """Tek GPU ile DeepSpeed training"""
    print(f"🔹 Tek GPU modu: DeepSpeed ZeRO-{ZERO_STAGE} ile eğitim")
    
    # Metrics collector başlat
    metrics = MetricsCollector(num_gpus=world)
    metrics.start()
    
    write_ds_config(use_pipeline=False)
    
    loader = DataLoader(
        train_ds,
        batch_size=SINGLE_GPU_BSZ,
        shuffle=True,
        collate_fn=collate_fn_single,
        drop_last=True
    )
    
    # Gradient checkpointing - bellek kullanımını azaltır
    peft_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    print("   └─ Gradient checkpointing aktif")
    
    engine, _, _, _ = deepspeed.initialize(
        model=peft_model,
        model_parameters=[p for p in peft_model.parameters() if p.requires_grad],
        config="ds_config.json"
    )
    
    # Toplam global step sayısı (transformers tarzı: optimizer step)
    total_batches = len(loader)
    total_global_steps = total_batches // SINGLE_GPU_GRAD_ACCUM
    
    for epoch in range(EPOCHS):
        engine.train()
        accumulated_loss = 0
        global_step = 0
        
        # tqdm progress bar - transformers style (global steps göster)
        progress_bar = tqdm(
            total=total_global_steps,
            desc=f"Epoch {epoch + 1}/{EPOCHS}",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
            dynamic_ncols=True
        )
        
        for batch_idx, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(engine.device)
            attention_mask = batch["attention_mask"].to(engine.device)
            labels = batch["labels"].to(engine.device)
            
            outputs = engine(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                use_cache=False
            )
            
            loss = outputs.loss
            
            # Loss skaler olduğundan emin ol
            if loss.dim() > 0:
                loss = loss.mean()
            
            # DeepSpeed otomatik gradient accumulation yapar
            engine.backward(loss)
            engine.step()
            
            loss_val = loss.item()
            accumulated_loss += loss_val
            
            # Her GRAD_ACCUM batch'te bir global step sayılır (transformers tarzı)
            if (batch_idx + 1) % SINGLE_GPU_GRAD_ACCUM == 0:
                global_step += 1
                avg_step_loss = accumulated_loss / SINGLE_GPU_GRAD_ACCUM
                
                # Metrics kaydet (optimizer step başına)
                metrics.log_step(epoch, global_step, avg_step_loss)
                
                # Progress bar güncelle
                progress_bar.update(1)
                progress_bar.set_postfix({
                    "loss": f"{avg_step_loss:.4f}",
                    "global_step": f"{global_step}/{total_global_steps}"
                })
                
                accumulated_loss = 0  # Reset for next accumulation
        
        progress_bar.close()
        print(f"\n✅ Epoch {epoch + 1} tamamlandı - Total Steps: {global_step}", flush=True)
    
    # Metrics kaydet
    metrics.save_csv()
    metrics.save_summary_json()
    
    # Modeli kaydet
    out_dir = os.path.join(SAVE_DIR, "hf_model")
    os.makedirs(out_dir, exist_ok=True)
    peft_model.save_pretrained(out_dir, safe_serialization=True)
    peft_model.base_model.config.save_pretrained(out_dir)
    print(f"✅ Model kaydedildi: {out_dir}", flush=True)

def train_multi_gpu_pipeline(peft_model, train_ds, rank, world):
    """Multi-GPU Pipeline Parallelism ile training"""
    print(f"🔹 Pipeline Parallelism modu: {world} GPU ile eğitim")
    
    # Metrics collector başlat (sadece rank 0'da kaydet)
    metrics = MetricsCollector(num_gpus=world) if rank == 0 else None
    if metrics:
        metrics.start()
    
    write_ds_config(use_pipeline=True)
    
    embed, blocks, final_norm, lm_head, rotary_emb = parse_decoder_only_parts(peft_model)
    
    layers = create_pipeline_stages(
        embed, blocks, final_norm, lm_head, rotary_emb, world
    )
    
    pipe = PipelineModule(
        layers=layers,
        num_stages=world,
        partition_method="uniform"
    )
    
    loader = DataLoader(
        train_ds,
        batch_size=MICRO_BSZ,
        shuffle=True,
        collate_fn=collate_fn_pipeline,
        drop_last=True
    )
    
    engine, _, _, _ = deepspeed.initialize(
        model=pipe,
        model_parameters=[p for p in pipe.parameters() if p.requires_grad],
        config="ds_config.json"
    )
    
    for epoch in range(EPOCHS):
        engine.train()
        data_iter = iter(loader)
        total_global_steps = len(loader) // GRAD_ACCUM  # Transformers tarzı global step

        # tqdm progress bar - sadece rank 0'da göster
        if rank == 0:
            progress_bar = tqdm(
                range(total_global_steps),
                desc=f"Epoch {epoch + 1}/{EPOCHS}",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
                dynamic_ncols=True
            )
        else:
            progress_bar = range(total_global_steps)
        
        total_loss = 0
        global_step = 0
        
        for step in progress_bar:
            loss = engine.train_batch(data_iter=data_iter)
            loss_val = loss.item()
            
            total_loss += loss_val
            global_step += 1
            
            # Metrics kaydet (sadece rank 0)
            if metrics:
                metrics.log_step(epoch, global_step, loss_val)

            # Progress bar güncelle (sadece rank 0)
            if rank == 0:
                progress_bar.set_postfix({
                    "loss": f"{loss_val:.4f}",
                    "global_step": f"{global_step}/{total_global_steps}"
                })

        if rank == 0:
            avg_loss = total_loss / max(global_step, 1)
            print(f"\n✅ Epoch {epoch + 1} tamamlandı - Avg Loss: {avg_loss:.4f}", flush=True)
    
    # Metrics kaydet (sadece rank 0)
    if metrics:
        metrics.save_csv()
        metrics.save_summary_json()

    # ================== HF EXPORT (DAĞINIK GPU'LARDAN TOPLAMA) ==================
    if rank == 0:
        print("📦 Pipeline kapatılıyor, ağırlıklar toplanıyor...", flush=True)

    torch.distributed.barrier()
    
    local_state = {}
    for name, param in pipe.named_parameters():
        if param.requires_grad:
            local_state[name] = param.detach().cpu()

    gathered_states = [None for _ in range(world)]
    torch.distributed.all_gather_object(gathered_states, local_state)

    if rank == 0:
        full_lora_state = {}
        for state in gathered_states:
            full_lora_state.update(state)
            
        print(f"📦 Toplanan parametre sayısı: {len(full_lora_state)}")

        converted_state_dict = {}
        sorted_keys = sorted(full_lora_state.keys())
        pipeline_indices = sorted(list(set([int(k.split(".")[1]) for k in sorted_keys if k.startswith("layers.")])))
        
        hf_layer_idx = 0
        for p_idx in pipeline_indices:
            p_keys = [k for k in sorted_keys if k.startswith(f"layers.{p_idx}.")]
            
            for pk in p_keys:
                parts = pk.split(".")
                suffix = ".".join(parts[3:]) 
                new_key = f"base_model.model.model.layers.{hf_layer_idx}.{suffix}"
                converted_state_dict[new_key] = full_lora_state[pk]
            
            hf_layer_idx += 1

        print(f"📦 Dönüştürülen anahtar sayısı: {len(converted_state_dict)}")
        
        load_result = peft_model.load_state_dict(converted_state_dict, strict=False)
        print(f"Load Result: {load_result}")

        out_dir = os.path.join(SAVE_DIR, "hf_model")
        os.makedirs(out_dir, exist_ok=True)

        peft_model.save_pretrained(out_dir, safe_serialization=True)
        peft_model.base_model.config.save_pretrained(out_dir)

        print(f"✅ HF model başarıyla kaydedildi: {out_dir}", flush=True)

    torch.distributed.barrier()

# ==================== MAIN ====================
def main():
    deepspeed.init_distributed()
    rank = torch.distributed.get_rank()
    world = torch.distributed.get_world_size()

    if rank == 0:
        print(f"🚀 {world} GPU ile DeepSpeed eğitimi başlatılıyor...")
        if world == 1:
            print("   └─ Mod: Tek GPU (ZeRO Optimization)")
        else:
            print("   └─ Mod: Pipeline Parallelism")

    base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16
    )

    peft_model = get_peft_model(
        base,
        LoraConfig(
            r=64,
            lora_alpha=64,
            lora_dropout=0.05,
            bias="none",
            target_modules=[
                "q_proj", "k_proj", "v_proj",
                "o_proj", "up_proj", "down_proj", "gate_proj"
            ],
            task_type="CAUSAL_LM"
        )
    )

    train_ds = load_from_disk(TOKENIZED_TRAIN_PATH)

    # GPU sayısına göre eğitim modu seç
    if world == 1:
        train_single_gpu(peft_model, train_ds, rank, world)
    else:
        train_multi_gpu_pipeline(peft_model, train_ds, rank, world)

    torch.distributed.destroy_process_group()

if __name__ == "__main__":
    main()
