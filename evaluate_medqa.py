import math
import time
import torch
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
ADAPTER_PATH = "./pp_ckpt/hf_model"
TOKENIZER_PATH = BASE_MODEL_NAME
DEVICE = "cuda:0"
DTYPE = torch.bfloat16
MAX_NEW_TOKENS = 50
SYSTEM_PROMPT = "Choose the correct option. Do NOT add any comment or explanation while answering."


# ---- Model loading copied from evaluate.py ----
def load_model():
    print(f"Loading base model: {BASE_MODEL_NAME}")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        dtype=DTYPE,
        device_map=DEVICE,
    )
    print(f"Loading LoRA adapter: {ADAPTER_PATH}")
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)

    if model.config.pad_token_id is None:
        model.config.pad_token_id = model.config.eos_token_id

    model.eval()
    return model


def load_tokenizer():
    print(f"Loading tokenizer: {TOKENIZER_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


# ---- Data prep from Inference.ipynb ----
def prepare_dataset(split: str = "test") -> Dataset:
    dataset_name = "openlifescienceai/medqa"
    dataset = load_dataset(dataset_name)

    def build_split(df):
        rows = []
        for _, row in df.iterrows():
            item = row["data"]
            options_text = "\n".join(f"{k}: {v}" for k, v in item["Options"].items())
            user_prompt = f"{item['Question']}\n\n{options_text}"
            assistant_answer = f"Answer: {item['Correct Option']}) {item['Options'][item['Correct Option']]}"

            rows.append(
                {
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                        {"role": "assistant", "content": assistant_answer},
                    ],
                    "question": item["Question"],
                    "options": item["Options"],
                    "correct_option": item["Correct Option"],
                }
            )
        return Dataset.from_list(rows)

    df = dataset[split].to_pandas().drop(columns=["subject_name"])
    return build_split(df)


# ---- Inference/evaluation ----
def run_eval(model, tokenizer, test_data: Dataset, output_csv: str = "llm_inference_results_medqa.csv"):
    results = []

    for i, sample in enumerate(tqdm(test_data, desc="Evaluating", unit="sample")):
        user_prompt = sample["messages"][1]["content"]
        assistant_answer = sample["messages"][2]["content"]

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_tensors="pt",
            add_generation_prompt=True,
        )
        encoded = {k: v.to(model.device) for k, v in encoded.items()}

        input_ids = encoded["input_ids"]
        attention_mask = encoded.get("attention_mask")
        generate_kwargs = {
            "input_ids": input_ids,
            "max_new_tokens": MAX_NEW_TOKENS,
            "do_sample": False,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if attention_mask is not None:
            generate_kwargs["attention_mask"] = attention_mask

        with torch.no_grad():
            outputs = model.generate(**generate_kwargs)

        generated_text = tokenizer.decode(
            outputs[0][input_ids.shape[1]:],
            skip_special_tokens=True,
        ).strip()

        correct_option = assistant_answer.split(":")[1].split(")")[0].strip().upper()

        full_ids = outputs[0].unsqueeze(0)
        labels = full_ids.clone()
        labels[:, :input_ids.shape[-1]] = -100

        with torch.no_grad():
            out_full = model(input_ids=full_ids, labels=labels)
            loss = out_full.loss.item()
            perplexity = math.exp(loss)

        results.append(
            {
                "id": i,
                "model_output": generated_text,
                "correct_option": correct_option,
                "loss": loss,
                "perplexity": perplexity,
            }
        )

        if (i + 1) % 50 == 0:
            # Compute running accuracy lazily to avoid expensive recompute each step
            tmp_df = pd.DataFrame(results)
            if not tmp_df.empty:
                tmp_df["predicted_choice"] = tmp_df["model_output"].str.extract(r"([A-D])")
                tmp_df["is_correct"] = tmp_df["predicted_choice"].str.upper() == tmp_df["correct_option"].str.upper()
                running_acc = tmp_df["is_correct"].mean()
            else:
                running_acc = float("nan")
            print(f"[{i + 1}/{len(test_data)}] Latest loss: {loss:.4f} | Acc so far: {running_acc:.4f}")

        time.sleep(0.05)

    df = pd.DataFrame(results)
    df["predicted_choice"] = df["model_output"].str.extract(r"([A-D])")
    df["is_correct"] = df["predicted_choice"].str.upper() == df["correct_option"].str.upper()
    accuracy = df["is_correct"].mean()

    df.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"Saved results to {output_csv}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Mean loss: {df.loss.mean():.4f}")
    print(f"Mean perplexity: {df.perplexity.mean():.4f}")
    return df, accuracy


if __name__ == "__main__":
    model = load_model()
    tokenizer = load_tokenizer()
    test_data = prepare_dataset("test")
    run_eval(model, tokenizer, test_data)
