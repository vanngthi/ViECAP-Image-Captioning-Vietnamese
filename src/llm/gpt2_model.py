import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class GPT2LanguageModel:
    def __init__(self, model_name: str, device: str = "cpu", search="greedy"):
        self.device = device
        self.search = search

        print(f"[GPT2] Loading {model_name} (search={search})")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def encode(self, text):
        """
        Unified encode:
            - If text is string  -> return 1D tensor (token ids)
            - If text is list    -> return dict for batching
        """
        # Case 1: single caption
        if isinstance(text, str):
            ids = self.tokenizer.encode(text)
            return torch.tensor(ids, dtype=torch.int64)

        # Case 2: list of sentences
        elif isinstance(text, list):
            return self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self.device)

        else:
            raise ValueError(f"Invalid input type for encode(): {type(text)}")

    @torch.no_grad()
    def generate(self, prompt: str, max_len: int = 30):
        """Generate với search method đã chọn."""
        inputs = self.encode([prompt])  # always batch of size 1

        search = self.search.lower()

        if search == "greedy":
            out = self.model.generate(
                **inputs,
                max_length=max_len,
                do_sample=False
            )

        elif search == "beam":
            out = self.model.generate(
                **inputs,
                max_length=max_len,
                num_beams=5,
                early_stopping=True
            )

        elif search == "topk":
            out = self.model.generate(
                **inputs,
                max_length=max_len,
                do_sample=True,
                top_k=50,
            )

        elif search == "topp":
            out = self.model.generate(
                **inputs,
                max_length=max_len,
                do_sample=True,
                top_p=0.9
            )

        elif search == "temperature":
            out = self.model.generate(
                **inputs,
                max_length=max_len,
                do_sample=True,
                temperature=0.7
            )

        elif search == "beam_sampling":
            out = self.model.generate(
                **inputs,
                max_length=max_len,
                num_beams=5,
                do_sample=True,
                top_p=0.9
            )

        else:
            raise ValueError(f"Unknown search mode: {self.search}")

        return self.tokenizer.decode(out[0], skip_special_tokens=True)
