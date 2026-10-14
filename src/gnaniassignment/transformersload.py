"""Standalone experiment with Gnani's local causal model.

The LLM enquiry pipeline uses Vercel AI Gateway. Run this module explicitly only when
the optional Transformers dependencies and model resources are available.
"""


def main() -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = "gnani/gnani-evon-v3.3-30B-A3B"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
    inputs = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Who are you?"}],
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=40)
    print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:]))


if __name__ == "__main__":
    main()
