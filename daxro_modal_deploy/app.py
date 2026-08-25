"""
DAXRO fine-tuned model, served on Modal as an OpenAI-compatible endpoint.

Why OpenAI-compatible: the DATORYX backend already knows how to talk to
OpenAI-shaped APIs (that's how Groq, Grok, and OpenAI itself are wired in
models/llm/providers.py). Making this endpoint match that same
/v1/chat/completions shape means the main app needs zero new client code -
just point DaxroRemoteProvider's base_url at this Modal deployment.

Deploy with:
    modal deploy app.py

First run downloads the base model into a persistent Modal Volume (a few
GB, one-time). Subsequent deploys/calls reuse it - no re-download.
"""

import modal

app = modal.App("daxro-inference")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers>=4.40.0",
        "peft>=0.11.0",
        "bitsandbytes>=0.43.0",
        "accelerate>=0.30.0",
        "fastapi",
    )
)

# Persistent volume so the ~2GB base model isn't re-downloaded on every
# cold start.
model_volume = modal.Volume.from_name("daxro-model-cache", create_if_missing=True)

# The LoRA adapter is small (~120MB) - baked into the image directly rather
# than needing its own volume. Copy your adapter files into ./adapter/
# next to this app.py before running `modal deploy`.
image = image.add_local_dir("adapter", remote_path="/adapter")

BASE_MODEL = "unsloth/Qwen2.5-3B-Instruct-bnb-4bit"


@app.cls(
    image=image,
    gpu="T4",
    volumes={"/root/.cache/huggingface": model_volume},
    scaledown_window=300,  # keep warm for 5 min after last request
)
class Daxro:
    @modal.enter()
    def load(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import PeftModel

        bnb_config = BitsAndBytesConfig(load_in_4bit=True)
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, quantization_config=bnb_config, device_map="auto",
        )
        self.model = PeftModel.from_pretrained(base, "/adapter")
        self.model.eval()
        self.tok = AutoTokenizer.from_pretrained("/adapter")
        self.torch = torch

    @modal.fastapi_endpoint(method="POST", docs=True)
    def chat_completions(self, body: dict):
        """OpenAI-shaped /v1/chat/completions - matches what
        OpenAICompatibleProvider in the DATORYX backend already sends."""
        messages = body.get("messages", [])
        temperature = body.get("temperature", 0.7)
        max_tokens = min(body.get("max_tokens", 400), 800)

        inputs = self.tok.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt",
        ).to(self.model.device)
        with self.torch.no_grad():
            out = self.model.generate(
                input_ids=inputs,
                max_new_tokens=max_tokens,
                temperature=max(temperature, 0.1),
                do_sample=True,
                pad_token_id=self.tok.eos_token_id,
            )
        completion = self.tok.decode(out[0][inputs.shape[-1]:], skip_special_tokens=True).strip()

        # OpenAI chat/completions response shape
        return {
            "id": "daxro-ft-1.0",
            "object": "chat.completion",
            "model": "daxro-ft-1.0",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": completion},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": int(inputs.shape[-1]),
                "completion_tokens": len(out[0]) - int(inputs.shape[-1]),
                "total_tokens": len(out[0]),
            },
        }
