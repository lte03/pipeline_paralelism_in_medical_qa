import os
import json
import random
import numpy as np
import pandas as pd
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from huggingface_hub import login
import torch
from transformers import set_seed

# ==================== GLOBAL AYARLAR ====================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
set_seed(SEED)

api = ""

# ==================== MODEL VE DATASET ====================
model_name = "Qwen/Qwen2.5-7B-Instruct"
login(api)

dataset_name = "openlifescienceai/medqa"
dataset = load_dataset(dataset_name)

# ==================== DATAFRAME DÖNÜŞÜM ====================
def prepare_dataset(df):
    new_rows = []
    for _, row in df.iterrows():
        item = row["data"]
        options_text = "\n".join([f"{k}: {v}" for k, v in item["Options"].items()])
        question_with_ops = f"{item['Question']}\n\n{options_text}"

        user_prompt = question_with_ops
        assistant_answer = f"Answer: {item['Correct Option']}) {item['Options'][item['Correct Option']]}"

        new_rows.append({
            "id": row["id"],
            "messages": [
                {"role": "system", "content": "Choose the correct option. Do NOT add any comment or explanation while answering."},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": assistant_answer}
            ],
            "question": item["Question"],
            "options": item["Options"],
            "correct_option": item["Correct Option"]
        })
    return Dataset.from_list(new_rows)

train_df = dataset["train"].to_pandas().drop("subject_name", axis=1)
dev_df = dataset["dev"].to_pandas().drop("subject_name", axis=1)
test_df = dataset["test"].to_pandas().drop("subject_name", axis=1)

train_data = prepare_dataset(train_df)
dev_data = prepare_dataset(dev_df)
test_data = prepare_dataset(test_df)

columns_to_remove = ["id", "question", "options", "correct_option"]
train_data = train_data.remove_columns(columns_to_remove)
dev_data = dev_data.remove_columns(columns_to_remove)
test_data = test_data.remove_columns(columns_to_remove)

# ==================== TOKENIZER ====================
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=False)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples, max_length=512):
    all_input_ids, all_attention_mask, all_labels = [], [], []

    for messages in examples["messages"]:
        # Assistant hariç mesajlar prompt için
        messages_without_assistant = [m for m in messages if m["role"] != "assistant"]

        if messages_without_assistant:
            prompt_text = tokenizer.apply_chat_template(
                messages_without_assistant,
                tokenize=False,
                add_generation_prompt=True
            )
            prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=True)
        else:
            prompt_tokens = []

        full_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        full_tokens = tokenizer.encode(full_text, add_special_tokens=True)

        prompt_length = len(prompt_tokens)

        # Uzun sekansları kırp
        if len(full_tokens) > max_length:
            full_tokens = full_tokens[:max_length]
            prompt_length = min(prompt_length, max_length)

        seq_len = len(full_tokens)
        pad_len = max_length - seq_len

        input_ids = full_tokens + [tokenizer.pad_token_id] * pad_len
        attention_mask = [1] * seq_len + [0] * pad_len

        labels = (
            [-100] * prompt_length +
            input_ids[prompt_length:seq_len] +
            [-100] * pad_len
        )

        all_input_ids.append(input_ids)
        all_attention_mask.append(attention_mask)
        all_labels.append(labels)

    return {
        "input_ids": all_input_ids,
        "attention_mask": all_attention_mask,
        "labels": all_labels
    }

MAX_LENGTH = 512

# ==================== TOKENIZATION ====================
print("Tokenizing train data...")
train_tokenized = train_data.map(
    lambda x: tokenize_function(x, max_length=MAX_LENGTH),
    batched=True,
    remove_columns=train_data.column_names,
    desc="Tokenizing train"
)

print("Tokenizing dev data...")
dev_tokenized = dev_data.map(
    lambda x: tokenize_function(x, max_length=MAX_LENGTH),
    batched=True,
    remove_columns=dev_data.column_names,
    desc="Tokenizing dev"
)

print("Tokenizing test data...")
test_tokenized = test_data.map(
    lambda x: tokenize_function(x, max_length=MAX_LENGTH),
    batched=True,
    remove_columns=test_data.column_names,
    desc="Tokenizing test"
)

# ==================== KAYDET ====================
os.makedirs("./tokenized_data", exist_ok=True)
train_tokenized.save_to_disk("./tokenized_data/train")
dev_tokenized.save_to_disk("./tokenized_data/dev")
test_tokenized.save_to_disk("./tokenized_data/test")
