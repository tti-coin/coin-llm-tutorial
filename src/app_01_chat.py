from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

MODEL_NAME_OR_PATH = "SakanaAI/TinySwallow-1.5B-Instruct"


def main():
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME_OR_PATH,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME_OR_PATH,
        device_map="auto",
    )
    model.eval()

    messages = [
        {
            "role": "system",
            "content": "You are a helpful, concise, and accurate AI assistant.",
        },
    ]

    print("Enter your prompt (Ctrl-C or Ctrl-D to exit):")
    while True:
        try:
            user_input = input("> ")
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        prompt_text = tokenizer.apply_chat_template(
            messages + [{"role": "user", "content": user_input}],
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = tokenizer(
            prompt_text,
            return_tensors="pt",
            add_special_tokens=False,
        ).to(model.device)
        outputs = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=1024,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            num_beams=1,
        )
        response = tokenizer.decode(
            outputs[0, inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        )

        print(response)


if __name__ == "__main__":
    main()
