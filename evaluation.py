import os
import json
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap

import torch
from PIL import Image
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoProcessor, AutoModel

# ===== IMPORT model và utils của bạn =====
from src.models.ClipCap import ClipCaptionModel
from src.data.utils import compose_discrete_prompts
from src.data.load_annotations import load_entities_text
from src.detector.detection import ObjectDetector
from src.data.search import greedy_search, beam_search
from src.data.retrieval_categories import clip_texts_embeddings, image_text_simiarlity, top_k_categories


# ====================================================
# 1. API INFERENCE ĐƠN GIẢN – GỌI ẢNH → TRẢ VỀ CAPTION
# ====================================================
@torch.no_grad()
def generate_caption(model, encoder, processor, tokenizer, args, image_path):
    device = args.device
    image = Image.open(image_path).convert("RGB")
    
    # Encode image
    inputs = processor(images=image, return_tensors="pt").to(device)
    image_features = encoder.get_image_features(**inputs)
    image_features = F.normalize(image_features, dim=-1)

    # Continuous prompt
    continuous_embeddings = model.mapping_network(image_features).view(
        -1, args.continuous_prompt_length, model.gpt_hidden_size
    )

    # Choose ORIGINAL or DETECT mode
    if args.using_hard_prompt:
        if args.mode == "original":
            entities_text = args.entities_text
            texts_embeddings = args.texts_embeddings

            logits = image_text_simiarlity(
                texts_embeddings,
                temperature=args.temperature,
                images_features=image_features
            )
            detected_objects, _ = top_k_categories(
                entities_text, logits, args.top_k, args.threshold
            )
            detected_objects = detected_objects[0]
        else:
            detected_objects = list(args.detector.detect(image))

        discrete_tokens = compose_discrete_prompts(tokenizer, detected_objects)
        discrete_tokens = discrete_tokens.unsqueeze(0).to(device)
        discrete_embeddings = model.word_embed(discrete_tokens)

        # merge
        if args.only_hard_prompt:
            embeddings = discrete_embeddings
        elif args.soft_prompt_first:
            embeddings = torch.cat((continuous_embeddings, discrete_embeddings), dim=1)
        else:
            embeddings = torch.cat((discrete_embeddings, continuous_embeddings), dim=1)
    else:
        embeddings = continuous_embeddings

    # BEAM or GREEDY
    if not args.using_greedy_search:
        out = beam_search(embeddings, tokenizer, args.beam_width, model.gpt)[0]
    else:
        out = greedy_search(embeddings, tokenizer, model.gpt)

    return out


# ====================================================
# 2. EVALUATION LOOP
# ====================================================
def evaluate_coco(args):

    coco = COCO(args.coco_gt_json)

    # Load model / processor / etc.
    tokenizer = AutoTokenizer.from_pretrained(args.language_model)
    processor = AutoProcessor.from_pretrained(args.clip_model)
    encoder = AutoModel.from_pretrained(args.clip_model).to(args.device).eval()

    model = ClipCaptionModel(
        continuous_length=args.continuous_prompt_length,
        clip_project_length=args.clip_project_length,
        clip_hidden_size=args.clip_hidden_size,
        gpt_type=args.language_model,
        num_layers=args.num_layers,
        soft_prompt_first=args.soft_prompt_first,
        only_hard_prompt=args.only_hard_prompt,
    )
    model.load_state_dict(
        torch.load(args.weight_path, map_location=args.device),
        strict=False
    )
    model.to(args.device).eval()

    # Preload entities if mode == "original"
    if args.mode == "original":
        args.entities_text = load_entities_text(
            args.name_of_entities_text,
            args.path_of_entities,
            not args.disable_all_entities
        )
        args.texts_embeddings = clip_texts_embeddings(
            args.entities_text, args.path_of_entities_embeddings
        )
    else:
        args.detector = ObjectDetector(args.detector_config, args.device)

    results = []

    img_ids = coco.getImgIds()
    for img_id in tqdm(img_ids):
        img_info = coco.loadImgs(img_id)[0]
        file_name = img_info['file_name']
        img_path = os.path.join(args.coco_images_dir, file_name)

        caption = generate_caption(
            model, encoder, processor, tokenizer, args, img_path
        )
        results.append({
            "image_id": img_id,
            "caption": caption
        })

    # Save results
    os.makedirs(args.save_dir, exist_ok=True)
    result_json = os.path.join(args.save_dir, "captions_results.json")
    with open(result_json, "w") as f:
        json.dump(results, f, indent=2)

    # Run COCO evaluation
    coco_res = coco.loadRes(result_json)

    coco_eval = COCOEvalCap(coco, coco_res)
    coco_eval.evaluate()

    print("\n===== COCO EVALUATION RESULTS =====")
    for metric, score in coco_eval.eval.items():
        print(f"{metric}: {score:.4f}")



# ====================================================
# MAIN
# ====================================================
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
    parser.add_argument("--weight_path", default="./checkpoints/coco_prefix-0014.pt")
    parser.add_argument("--using_hard_prompt", action="store_true")
    parser.add_argument("--soft_prompt_first", action="store_true")
    parser.add_argument("--only_hard_prompt", action="store_true")
    parser.add_argument("--using_greedy_search", action="store_true")
    parser.add_argument("--beam_width", type=int, default=5)

    # entity / detection
    parser.add_argument("--mode", default="original")
    parser.add_argument("--path_of_entities", default="./src/config/vietnamese_entities.json")
    parser.add_argument("--path_of_entities_embeddings", default="./dataset/entities_embeddings.pickle")
    parser.add_argument("--name_of_entities_text", default="vietnamese_entities")
    parser.add_argument("--detector_config", default="./src/config/detector.yaml")
    parser.add_argument("--disable_all_entities", action="store_true")

    # COCO evaluation
    parser.add_argument("--coco_images_dir", default="./coco/val2017/")
    parser.add_argument("--coco_gt_json", default="./coco/annotations/captions_val2017.json")
    parser.add_argument("--save_dir", default="./eval_results/")

    args = parser.parse_args()

    evaluate_coco(args)
