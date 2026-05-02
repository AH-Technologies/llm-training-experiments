"""Utility functions for SFT data generation."""


def format_prompt(question, tokenizer, use_chat_template=True):
    """Format a math question as token ids."""
    if not isinstance(question, str):
        messages = list(question)
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return tokenizer.encode(text)
    if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": question}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return tokenizer.encode(text)
    else:
        return tokenizer.encode(question)
