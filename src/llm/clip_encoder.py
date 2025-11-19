import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoProcessor, AutoModel

class CLIPEncoder:
    """
        Wrapper cho AltCLIP/CLIP để dùng encode image + text.
    """

    def __init__(self, model_name: str, device: str = "cpu"):
        print(f"[CLIPEncoder] Loading model {model_name} on {device}")
        self.device = device

        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device).eval()
        self.hidden_size = self.model.text_projection.weight.shape[1]

        print(f"[CLIPEncoder] hidden_size = {self.hidden_size}")

    @torch.no_grad()
    def encode_image(self, image):
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        feats = self.model.get_image_features(**inputs)
        return F.normalize(feats, dim=-1)

    @torch.no_grad()
    def encode_text(self, texts):
        inputs = self.processor(text=texts,
                                return_tensors="pt",
                                padding=True,
                                truncation=True).to(self.device)
        feats = self.model.get_text_features(**inputs)
        return F.normalize(feats, dim=-1)

    @torch.no_grad()
    def similarity(self, img_feats, txt_feats):
        """Cosine similarity between encoded features."""
        return (img_feats @ txt_feats.T).cpu()
