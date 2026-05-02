"""Debug: understand model predictions on arithmetic prompts."""
import torch
from transformer_lens import HookedTransformer

model = HookedTransformer.from_pretrained(
    "Qwen/Qwen2.5-1.5B-Instruct", device="cuda", dtype=torch.float16,
)

# Test different prompt formats
prompts = [
    "3 + 5 =",
    "3 + 5 = ",       # trailing space
    "What is 3 + 5?",
    "3+5=",
    "Calculate: 3 + 5 =",
]

print("=== Raw prompt format exploration ===\n")
for prompt in prompts:
    tokens = model.to_tokens(prompt)
    logits = model(tokens)
    top5 = logits[0, -1].topk(5)
    print(f"Prompt: '{prompt}' (len={tokens.shape[1]})")
    for i, (val, idx) in enumerate(zip(top5.values, top5.indices)):
        tok_str = model.to_string(idx.unsqueeze(0))
        print(f"  #{i+1}: '{tok_str}' (id={idx.item()}, logit={val.item():.2f})")
    print()

# Check: after the space, does the model predict the right number?
print("=== Two-step prediction (space then number) ===\n")
for a, b in [(3, 5), (7, 2), (1, 9), (15, 4)]:
    prompt = f"{a} + {b} = "  # include the space
    tokens = model.to_tokens(prompt)
    logits = model(tokens)
    top5 = logits[0, -1].topk(5)
    expected = a + b
    print(f"'{a} + {b} = ' -> expected {expected}")
    for i, (val, idx) in enumerate(zip(top5.values, top5.indices)):
        tok_str = model.to_string(idx.unsqueeze(0))
        print(f"  #{i+1}: '{tok_str}' (id={idx.item()}, logit={val.item():.2f})")
    print()

# Try chat template
print("=== Chat template format ===\n")
try:
    # Qwen chat format
    chat_prompt = "<|im_start|>user\nWhat is 3 + 5?<|im_end|>\n<|im_start|>assistant\n"
    tokens = model.to_tokens(chat_prompt)
    logits = model(tokens)
    top5 = logits[0, -1].topk(5)
    print(f"Chat prompt (len={tokens.shape[1]})")
    for i, (val, idx) in enumerate(zip(top5.values, top5.indices)):
        tok_str = model.to_string(idx.unsqueeze(0))
        print(f"  #{i+1}: '{tok_str}' (id={idx.item()}, logit={val.item():.2f})")
except Exception as e:
    print(f"Chat template failed: {e}")

print()

# Try: prompt ends with "= " so model predicts the number directly
print("=== Accuracy with trailing space (answer is first token) ===\n")
import random
random.seed(42)
correct = 0
total = 50
for _ in range(total):
    a, b = random.randint(1, 9), random.randint(1, 9)
    ans = a + b
    prompt = f"{a} + {b} = "
    tokens = model.to_tokens(prompt)
    logits = model(tokens)
    pred_id = logits[0, -1].argmax().item()
    pred_str = model.to_string(torch.tensor([pred_id]))
    ans_id = model.to_tokens(str(ans))[0, -1].item()  # no space prefix
    is_correct = pred_id == ans_id
    if is_correct:
        correct += 1
    elif _ < 5:
        print(f"  {a} + {b} = {ans}, predicted '{pred_str}' (id={pred_id}), expected id={ans_id}")

print(f"Accuracy with trailing space: {correct}/{total} = {correct/total:.1%}")
