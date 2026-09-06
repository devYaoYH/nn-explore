import torch
from nnsight import LanguageModel

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

model = LanguageModel(MODEL_NAME, device_map="auto", torch_dtype=torch.bfloat16)

print(f"Number of layers: {len(model.model.layers)}")
print(f"Hidden size: {model.config.hidden_size}")
print(f"Layer 0 structure:\n{model.model.layers[0]}")

with model.trace("The capital of France is"):
    # NOTE: on transformers 5.x, decoder layer .output is the hidden_states
    # tensor directly ([batch, seq, hidden]), not a (tensor,) tuple like in
    # older transformers versions. No [0] indexing needed.
    out = model.model.layers[5].output[:, -1, :].save()

print("Captured activation shape:", out.shape)
print("Smoke test passed.")
