import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoProcessor, AutoModel
from src.models.ClipCap import ClipCaptionModel

class CLIPEncoder:
    """Wrapper cho AltCLIP (BAAI/AltCLIP-m18 hoặc tương tự)."""

    def __init__(self, model_name: str, device: str):
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device).eval()
        self.device = device

    @torch.no_grad()
    def encode_image(self, image):
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        feats = self.model.get_image_features(**inputs)
        return F.normalize(feats, dim=-1)

    @torch.no_grad()
    def encode_text(self, texts):
        inputs = self.processor(text=texts, return_tensors="pt", padding=True).to(self.device)
        feats = self.model.get_text_features(**inputs)
        return F.normalize(feats, dim=-1)

class GPT2Decoder:
    """Wrapper cho GPT2 tiếng Việt (NlpHUST/gpt2-vietnamese)."""

    def __init__(self, model_name: str, device: str):
        from transformers import AutoTokenizer, GPT2LMHeadModel
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name).to(device).eval()
        self.device = device

    def encode_prompt(self, prompt: str):
        return torch.tensor(self.tokenizer.encode(prompt), dtype=torch.long).unsqueeze(0).to(self.device)

    def decode_tokens(self, tokens: torch.Tensor):
        return self.tokenizer.decode(tokens.squeeze().tolist(), skip_special_tokens=True)



def load_all_models(language_model, clip_model, clip_hidden, weight_path, device):
    """Tiện ích khởi tạo tất cả model chính."""
    clip_encoder = CLIPEncoder(clip_model, device)
    caption_model = ClipCapModel(clip_hidden, language_model, weight_path, device)
    tokenizer = AutoTokenizer.from_pretrained(language_model)
    return clip_encoder, caption_model, tokenizer
