import os
import pickle
import torch
import argparse
from transformers import AutoModel, AutoProcessor
import torch.nn.functional as F
from tqdm import tqdm


@torch.no_grad()
def embed(device: str, model_name: str, inpath: str, outpath: str):
    """
    Extract text embeddings using BAAI/AltCLIP-m18.
    Each caption (string) → 1024-dim normalized embedding tensor.
    """
    print(f"Loading AltCLIP model: {model_name} on {device}")
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()

    # Load captions [[entities, caption], ...]
    with open(inpath, 'rb') as infile:
        captions_with_entities = pickle.load(infile)

    print(f"Encoding {len(captions_with_entities)} captions...")
    for idx in tqdm(range(len(captions_with_entities))):
        caption = captions_with_entities[idx][1]
        inputs = processor(text=caption, return_tensors="pt", padding=True, truncation=True).to(device)
        text_features = model.get_text_features(**inputs)
        text_features = F.normalize(text_features, dim=-1).cpu()[0]
        captions_with_entities[idx].append(text_features)

    # Save
    with open(outpath, 'wb') as outfile:
        pickle.dump(captions_with_entities, outfile)

    print(f"Saved embeddings → {outpath}")
    return captions_with_entities


if __name__ == '__main__':
    print("Text Features Extracting ...", flush=True)
    parser = argparse.ArgumentParser(description="Extract text embeddings using AltCLIP")
    parser.add_argument('--inpath', type=str, required=True, help="Input pickle file (with captions + entities)")
    parser.add_argument('--outpath', type=str, required=True, help="Output pickle with embeddings")
    args = parser.parse_args()

    # Device setup
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model_name = "BAAI/AltCLIP-m18"

    # Run
    if os.path.exists(args.outpath):
        print(f"Found existing {args.outpath}, loading...")
        with open(args.outpath, 'rb') as infile:
            captions_with_features = pickle.load(infile)
    else:
        captions_with_features = embed(device, model_name, args.inpath, args.outpath)

    # Quick check
    import random
    print(f"\nDataset: {args.inpath}")
    print(f"Number of samples: {len(captions_with_features)}")

    sample = random.choice(captions_with_features)
    detected_entities, caption, caption_features = sample
    print(f"Entities: {detected_entities}")
    print(f"Caption:  {caption}")
    print(f"Feature shape: {tuple(caption_features.shape)}, dtype={caption_features.dtype}")

    # Check encode consistency
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    with torch.no_grad():
        inputs = processor(text=caption, return_tensors="pt", padding=True, truncation=True).to(device)
        new_emb = F.normalize(model.get_text_features(**inputs), dim=-1).cpu()[0]

    diff = torch.abs(new_emb - caption_features).mean().item()
    print(f"Embedding difference mean: {diff:.6f}")