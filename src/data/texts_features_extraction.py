import os
import pickle
import torch
import argparse
import random
from tqdm import tqdm
import torch.nn.functional as F

from src.llm.clip_encoder import CLIPEncoder


@torch.no_grad()
def embed(encoder: CLIPEncoder, inpath: str, outpath: str):
    """
    Extract text embeddings for captions.
    Input pickle format: [ [entities, caption], ... ]
    Output adds text embedding as index 2.
    """
    # Load captions
    with open(inpath, "rb") as infile:
        captions_with_entities = pickle.load(infile)

    print(f"Encoding {len(captions_with_entities)} captions...")

    for idx in tqdm(range(len(captions_with_entities))):
        caption = captions_with_entities[idx][1]
        feats = encoder.encode_text([caption])[0].cpu()
        captions_with_entities[idx].append(feats)

    with open(outpath, "wb") as outfile:
        pickle.dump(captions_with_entities, outfile)

    print(f"Saved embeddings → {outpath}")
    return captions_with_entities


if __name__ == "__main__":
    print("Text Features Extracting ...", flush=True)

    parser = argparse.ArgumentParser(description="Extract text embeddings using AltCLIP/CLIP")
    parser.add_argument("--inpath", type=str, required=True, help="Pickle chứa captions + entities")
    parser.add_argument("--outpath", type=str, required=True, help="Pickle output")
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model_name = "BAAI/AltCLIP-m18"

    # Load encoder 1 lần duy nhất
    encoder = CLIPEncoder(model_name=model_name, device=device)

    # Nếu file output có rồi thì load
    if os.path.exists(args.outpath):
        print(f"Found existing {args.outpath}, loading...")
        with open(args.outpath, "rb") as infile:
            captions_with_features = pickle.load(infile)
    else:
        captions_with_features = embed(encoder, args.inpath, args.outpath)

    # ======================
    # Quick check
    # ======================
    print(f"\nDataset: {args.inpath}")
    print(f"Samples: {len(captions_with_features)}")

    sample = random.choice(captions_with_features)
    entities, caption, features = sample

    print(f"Entities: {entities}")
    print(f"Caption:  {caption}")
    print(f"Feature shape: {tuple(features.shape)}")

    # Consistency check
    with torch.no_grad():
        new_emb = encoder.encode_text([caption])[0].cpu()

    diff = torch.abs(new_emb - features).mean().item()
    print(f"Embedding difference mean: {diff:.6f}")
