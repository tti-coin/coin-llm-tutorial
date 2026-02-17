"""
trlを用いたSFTのサンプルコードです
"""

import os
import time
from datetime import datetime

import torch
import wandb
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    set_seed,
)
from trl import SFTTrainer, SFTConfig

MODEL_NAME_OR_PATH = "llm-jp/llm-jp-3-980m-instruct2"
DATASET_NAME = "sbintuitions/JCommonsenseQA"
SEED = 42


def main():
    set_seed(SEED)
    wandb.init()

    # トークナイザの読み込み
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(tokenizer)

    # モデルの読み込みとQLoRAの適用
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME_OR_PATH,
        dtype=torch.float16,
        quantization_config=quantization_config,
        device_map="auto",
    )
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, peft_config)
    print(model)
    model.print_trainable_parameters()

    # データセットの読み込み
    dataset = load_dataset(DATASET_NAME, split="train")
    print(f"Dataset loaded:\n{dataset}")
    dataset = dataset.shuffle(seed=SEED)
    dataset = dataset.select(range(1200))
    dataset = dataset.train_test_split(test_size=200, seed=SEED)
    ds_train = dataset["train"]
    ds_dev = dataset["test"]
    print(f"Dataset\n(train):\n{ds_train}\n(dev):\n{ds_dev}")

    # 学習用に事例を整形 詳細はこちらを参照 https://huggingface.co/docs/trl/sft_trainer
    def dataset_preprocess(example: dict) -> dict:
        choices = [
            example["choice0"],
            example["choice1"],
            example["choice2"],
            example["choice3"],
            example["choice4"],
        ]
        label = int(example["label"])
        result = {
            "prompt": [
                {
                    "role": "user",
                    "content": f'{example["question"]}',
                },
            ],
            "completion": [
                {
                    "role": "assistant",
                    "content": f"{choices[label]}です。知らんけど。(^_-)\n",
                },
            ],
        }
        return result

    ds_train = ds_train.map(dataset_preprocess, remove_columns=ds_train.column_names)
    ds_dev = ds_dev.map(dataset_preprocess, remove_columns=ds_dev.column_names)

    # 学習の設定
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR + "/model",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,  # batch_size * gradient_accumulation_steps = 実効バッチサイズ
        num_train_epochs=1,
        learning_rate=1e-4,
        lr_scheduler_type="constant",
        logging_steps=10,
        do_eval=True,
        eval_strategy="epoch",
        eval_on_start=True,
        save_strategy="epoch",
        report_to="wandb",
    )
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=ds_train,
        eval_dataset=ds_dev,
    )

    # 学習の実行とモデルの保存
    trainer.train()
    trainer.save_model(OUTPUT_DIR + "/model")


if __name__ == "__main__":
    global OUTPUT_DIR
    for _ in range(5):
        OUTPUT_DIR = f"./outputs/{datetime.now().strftime('%Y-%m-%d/%H-%M-%S')}"
        try:
            os.makedirs(OUTPUT_DIR, exist_ok=False)
            print(f"OUTPUT_DIR: {OUTPUT_DIR}")
            break
        except FileExistsError:
            print(f"Output directory already exists: {OUTPUT_DIR}. Retrying...")
            time.sleep(1)
    else:
        raise RuntimeError("Failed to create a output directory.")

    main()
