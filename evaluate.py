import os
import json
import torch
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_from_disk
from peft import PeftModel
from tqdm import tqdm
import torch.nn.functional as F

# ==================== AYARLAR ====================
BASE_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
ADAPTER_PATH = "./pp_ckpt/hf_model"
TOKENIZER_PATH = BASE_MODEL_NAME # Tokenizer kaydedilmediği için base'den çekiyoruz
TOKENIZED_TEST_PATH = "./tokenized_data/test"

DEVICE = "cuda:0"
DTYPE = torch.bfloat16 # Training ile uyumlu
MAX_NEW_TOKENS = 8


# ==================== MODEL ====================
def load_model():
    print(f"🤖 Base Model: {BASE_MODEL_NAME}")

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=DTYPE,
        device_map=DEVICE
    )
    
    print(f"🔗 LoRA Adapter Yükleniyor: {ADAPTER_PATH}")
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)

    if model.config.pad_token_id is None:
        model.config.pad_token_id = model.config.eos_token_id

    model.eval()
    return model


def load_tokenizer():
    print(f"📝 Tokenizer yükleniyor: {TOKENIZER_PATH}")

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


# ==================== INFERENCE ====================
def evaluate_from_tokenized(n=None):
    model = load_model()
    tokenizer = load_tokenizer()

    dataset = load_from_disk(TOKENIZED_TEST_PATH)
    if n is not None:
        dataset = dataset.select(range(min(n, len(dataset))))

    eos_ids = [tokenizer.eos_token_id]
    if "<|eot_id|>" in tokenizer.get_vocab():
        eos_ids.append(tokenizer.convert_tokens_to_ids("<|eot_id|>"))

    correct = 0
    total = 0
    total_loss = 0.0
    total_tokens = 0

    print(f"\n🚀 Base model inference | Örnek sayısı: {len(dataset)}\n")

    for i, row in enumerate(tqdm(dataset, desc="Evaluating")):
        input_ids = row["input_ids"]
        attention_mask = row["attention_mask"]
        labels = row["labels"]

        # 🔒 Label leakage yok
        try:
            answer_start = next(idx for idx, t in enumerate(labels) if t != -100)
        except StopIteration:
            continue

        prompt_ids = torch.tensor(
            input_ids[:answer_start],
            device=DEVICE
        ).unsqueeze(0)

        prompt_mask = torch.tensor(
            attention_mask[:answer_start],
            device=DEVICE
        ).unsqueeze(0)

        # === Loss hesaplaması için tam sequence ===
        full_input_ids = torch.tensor(input_ids, device=DEVICE).unsqueeze(0)
        full_labels = torch.tensor(labels, device=DEVICE).unsqueeze(0)

        with torch.no_grad():
            # Loss hesapla
            loss_outputs = model(
                input_ids=full_input_ids,
                labels=full_labels
            )
            
            # Token başına loss hesabı için
            valid_labels = full_labels[full_labels != -100]
            num_valid_tokens = valid_labels.numel()
            
            if num_valid_tokens > 0:
                total_loss += loss_outputs.loss.item() * num_valid_tokens
                total_tokens += num_valid_tokens

            # Generation
            outputs = model.generate(
                input_ids=prompt_ids,
                attention_mask=prompt_mask,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                temperature=0.0,
                eos_token_id=eos_ids,
                pad_token_id=tokenizer.pad_token_id
            )

        # === Prediction ===
        gen_ids = outputs[0][prompt_ids.shape[1]:]
        pred_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        # === Gold ===
        gold_ids = [t for t in labels if t != -100]
        gold_text = tokenizer.decode(gold_ids, skip_special_tokens=True).strip()

        # === Güvenli A/B/C çıkarımı ===
        def extract_choice(text):
            for ch in text.strip():
                if ch.upper() in ["A", "B", "C"]:
                    return ch.upper()
            return None

        pred_char = extract_choice(pred_text)
        gold_char = extract_choice(gold_text)

        is_correct = (pred_char is not None) and (pred_char == gold_char)

        if is_correct:
            correct += 1
        total += 1

        if i < 5:
            print("\n" + "=" * 60)
            print(f"🧪 Örnek #{i+1}")
            print("=" * 60)
            print(f"🤖 Tahmin raw : {repr(pred_text)}")
            print(f"🤖 Tahmin     : {pred_char}")
            print(f"✅ Gerçek     : {gold_char}")
            print(f"📊 Sonuç      : {'DOĞRU' if is_correct else 'YANLIŞ'}")

    # === Metrikleri hesapla ===
    acc = 100 * correct / total if total > 0 else 0
    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
    perplexity = torch.exp(torch.tensor(avg_loss)).item() if total_tokens > 0 else float('inf')

    print("\n" + "=" * 60)
    print("🎯 FINAL SONUÇ (BASE MODEL)")
    print("=" * 60)
    print(f"Toplam     : {total}")
    print(f"Doğru      : {correct}")
    print(f"Yanlış     : {total - correct}")
    print(f"Accuracy   : %{acc:.2f}")
    print(f"Loss       : {avg_loss:.4f}")
    print(f"Perplexity : {perplexity:.4f}")
    print("=" * 60)

    # === Metrikleri dosyaya kaydet ===
    metrics = {
        "timestamp": datetime.now().isoformat(),
        "model": BASE_MODEL_NAME,
        "adapter_path": ADAPTER_PATH,
        "total_samples": total,
        "correct": correct,
        "wrong": total - correct,
        "accuracy": round(acc, 4),
        "loss": round(avg_loss, 6),
        "perplexity": round(perplexity, 4),
        "total_tokens": total_tokens
    }
    
    # Sonuçları JSON dosyasına kaydet
    output_dir = "./evaluation_results"
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"eval_metrics_{timestamp_str}.json")
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)
    
    print(f"\n📁 Metrikler kaydedildi: {output_file}")
    
    return metrics


if __name__ == "__main__":
    evaluate_from_tokenized(n=None)

