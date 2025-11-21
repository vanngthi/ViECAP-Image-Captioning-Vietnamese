
import os
import json
import pickle
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor
import torch.nn.functional as F

@torch.no_grad()
def generate_ensemble_prompt_embeddings(model_name, device, entities, outpath):
    # nếu đã tồn tại → load lại
    if os.path.exists(outpath):
        print(f"[OK] Found existing embeddings at {outpath}")
        with open(outpath, "rb") as f:
            return pickle.load(f)

    print(f"[Load] AltCLIP model: {model_name}")
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()

    all_embeddings = []

    for text in tqdm(entities, desc="Encoding Vietnamese entities"):
        inputs = processor(
            text=[text],
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(device)

        text_feat = model.get_text_features(**inputs).squeeze(0)
        text_feat = F.normalize(text_feat, dim=-1)
        all_embeddings.append(text_feat.cpu())

    all_embeddings = torch.stack(all_embeddings, dim=0)

    with open(outpath, "wb") as f:
        pickle.dump(all_embeddings, f)

    print(f"[Saved] embeddings → {outpath}")
    return all_embeddings


if __name__ == "__main__":
    # load vocab tiếng Việt
    with open("../config/vietnamese_entities.json", "r", encoding="utf-8") as f:
        entities = json.load(f)

    outpath = "/DATA/van-n/phenikaa/ViTrCap/dataset/vietnamese_entities_embeddings.pickle"

    embeddings = generate_ensemble_prompt_embeddings(
        model_name="BAAI/AltCLIP-m18",
        device="cpu",
        entities=entities,
        outpath=outpath
    )

    print("Số lượng entity:", len(entities))
    print("Embedding shape:", embeddings.shape)   # (num_entities, 1024)
