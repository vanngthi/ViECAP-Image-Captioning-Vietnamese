import os
import json
import torch
from tqdm import tqdm
from PIL import Image
from transformers import AutoTokenizer, AutoModel, AutoProcessor
import torch.nn.functional as F

from src.models.ClipCap import ClipCaptionModel
from src.data.search import greedy_search, beam_search
from src.data.utils import compose_discrete_prompts

# COCO-caption metrics
from pycocoevalcap.eval import COCOEvalCap
from pycocotools.coco import COCO


@torch.no_grad()
def generate_caption(model, tokenizer, encoder, processor, image_path, device, args):
    image = Image.open(image_path).convert("RGB")

    inputs = processor(images=image, return_tensors="pt").to(device)
    image_features = encoder.get_image_features(**inputs)
    image_features = F.normalize(image_features, dim=-1)

    continuous_embeddings = model.mapping_network(image_features)
    continuous_embeddings = continuous_embeddings.view(
        -1, args.continuous_prompt_length, model.gpt_hidden_size
    )

    # Only soft prompt for evaluation
    embeddings = continuous_embeddings

    if args.greedy:
        sentence = greedy_search(
            embeddings=embeddings, tokenizer=tokenizer, model=model.gpt
        )
    else:
        result = beam_search(
            embeddings=embeddings,
            tokenizer=tokenizer,
            model=model.gpt,
            beam_width=args.beam_width,
        )
        sentence = result[0]

    # cleanup repetition artifacts
    sentence = sentence.replace("longlong", "")
    sentence = sentence.strip()

    return sentence


def evaluate(args):
    device = args.device

    # Load tokenizer + ClipCap
    tokenizer = AutoTokenizer.from_pretrained(args.language_model)
    model = ClipCaptionModel(
        continuous_length=args.continuous_prompt_length,
        clip_project_length=args.clip_project_length,
        clip_hidden_size=args.clip_hidden_size,
        gpt_type=args.language_model,
        num_layers=args.num_layers,
    )
    model.load_state_dict(torch.load(args.weight_path, map_location=device), strict=False)
    model.to(device).eval()

    # Load CLIP encoder
    processor = AutoProcessor.from_pretrained(args.clip_model)
    encoder = AutoModel.from_pretrained(args.clip_model).to(device).eval()

    # COCO formatted JSON
    results = []
    test_images = sorted(os.listdir(args.test_image_dir))

    for img_name in tqdm(test_images, desc="Evaluating"):
        image_path = os.path.join(args.test_image_dir, img_name)
        caption = generate_caption(
            model, tokenizer, encoder, processor, image_path, device, args
        )

        img_id = int(os.path.splitext(img_name)[0])  # image name = 12345.jpg

        results.append({"image_id": img_id, "caption": caption})

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    result_path = os.path.join(args.output_dir, "captions_results.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("Saved:", result_path)

    # =============== COCO EVAL ==================
    coco = COCO(args.coco_annotation_file)
    coco_result = coco.loadRes(result_path)
    coco_eval = COCOEvalCap(coco, coco_result)

    coco_eval.evaluate()

    print("\n===== Final Evaluation Metrics =====")
    for metric, score in coco_eval.eval.items():
        print(f"{metric}: {score:.4f}")

    return coco_eval.eval


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")

    parser.add_argument("--clip_model", default="ViT-B/32")
    parser.add_argument("--language_model", default="gpt2")

    parser.add_argument("--continuous_prompt_length", type=int, default=10)
    parser.add_argument("--clip_project_length", type=int, default=10)
    parser.add_argument("--clip_hidden_size", type=int, default=1024)
    parser.add_argument("--num_layers", type=int, default=8)

    # decoding
    parser.add_argument("--greedy", action="store_true", default=False)
    parser.add_argument("--beam_width", type=int, default=5)

    # paths
    parser.add_argument("--test_image_dir", type=str, required=True)
    parser.add_argument("--coco_annotation_file", type=str, required=True)
    parser.add_argument("--weight_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./eval_output")

    args = parser.parse_args()
    evaluate(args)
