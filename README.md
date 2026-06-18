<div align="center">

# **TreeFlash: Parallel AR-Approximation for Faster Speculative Decoding**

**Peer Rheinboldt · Frédéric Berdoz · Roger Wattenhofer**

[![arXiv](https://img.shields.io/badge/arXiv-2606.03819-b31b1b.svg)](https://arxiv.org/abs/2606.03819)

Preprint, submitted June 2026

</div>

---

https://github.com/user-attachments/assets/d53dbbc0-351d-4dfc-bc28-91dc67ad5aa9

---

## Quick Start

TreeFlash requires `trust_remote_code=True` because the drafter architecture and
`spec_generate` method are provided by this repository.

```python
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

drafter = AutoModel.from_pretrained(
    "peerrh/treeflash-qwen3-4b",
    trust_remote_code=True,
    dtype="bfloat16",
    device_map="cuda:0",
).eval()

target = AutoModelForCausalLM.from_pretrained(
    "qwen/qwen3-4b",
    trust_remote_code=True,
    dtype="bfloat16",
    device_map="cuda:0",
).eval()

tokenizer = AutoTokenizer.from_pretrained("qweb/qwen3-4b", trust_remote_code=True)

messages = [{"role": "user", "content": "Write a quicksort in Python."}]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,
)
inputs = tokenizer([text], return_tensors="pt").to(drafter.device)

output_ids = drafter.spec_generate(
    target=target,
    input_ids=inputs["input_ids"],
    max_new_tokens=2048,
    stop_token_ids=[tokenizer.eos_token_id],
    temperature=0.0,
    drafter_temperature=1.0,
    tree_size=64,
    top_m=16,
)

print(tokenizer.decode(output_ids[0], skip_special_tokens=True))
```

---

## Supported Models
| Target | Drafter |
| ---- | ------ |
| [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) | [peerrh/treeflash-qwen3-4b](https://huggingface.co/peerrh/treeflash-qwen3-4b) |
|[Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) | [peerrh/treeflash-qwen3-8b](https://huggingface.co/peerrh/treeflash-qwen3-8b) |
|[Qwen/Qwen3-Coder-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct) | [peerrh/treeflash-qwen3-coder-30b-a3b](https://huggingface.co/peerrh/treeflash-qwen3-coder-30b-a3b) |
---

## Citation
If you use TreeFlash, please cite:

```bibtex
@article{rheinboldt2026treeflash,
  title={TreeFlash: Parallel AR-Approximation for Faster Speculative Decoding},
  author={Rheinboldt, Peer and Berdoz, Fr{\'e}d{\'e}ric and Wattenhofer, Roger},
  journal={arXiv preprint arXiv:2606.03819},
  year={2026}
}
```
