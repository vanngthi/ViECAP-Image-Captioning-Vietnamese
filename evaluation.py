import argparse
import pickle
import csv
import torch
from tqdm import tqdm

from transformers import AutoTokenizer

from src.models.ClipCap import ClipCaptionModel
from src.data.search import greedy_search, beam_search, opt_search
from src.data.utils import compose_discrete_prompts


# ============================================================
#   STEP 1 — Generate captions with BATCH soft-prompt
# ============================================================
@torch.no_grad()
def generate_predictions(args):

    # ==== Load dataset ====
    dataset = pickle.load(open(args.pickle, "rb"))
    print(f"[Loaded pickle] {len(dataset)} samples")

    # ==== Load tokenizer + caption model ====
    tokenizer = AutoTokenizer.from_pretrained(args.language_model)

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

    # ==== Prepare CSV (prediction only) ====
    with open(args.out_csv, "w", newline="", encoding="utf8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)   # prevent comma split
        writer.writerow(["image_id", "pred_caption"])

        B = args.batch_size

        # ==== Loop batches ====
        for idx in tqdm(range(0, len(dataset), B), desc="Batch inference"):
            batch_items = dataset[idx: idx + B]
            batch_len = len(batch_items)

            # ======= 1) Batch soft prompt =======
            image_features_batch = torch.stack([
                torch.tensor(item["image_features"])
                for item in batch_items
            ]).to(args.device)

            continuous_embeddings = model.mapping_network(image_features_batch)
            continuous_embeddings = continuous_embeddings.view(
                batch_len,
                args.continuous_prompt_length,
                model.gpt_hidden_size
            )

            # ======= 2) Generate per item =======
            for bi in range(batch_len):
                item = batch_items[bi]
                ce = continuous_embeddings[bi:bi+1]  # (1, L, H)

                # ---- Hard prompt ----
                entities = item["entities"]
                if args.using_hard_prompt and len(entities) > 0:

                    discrete_tokens = compose_discrete_prompts(
                        tokenizer,
                        entities
                    ).unsqueeze(0).to(args.device)

                    discrete_embeddings = model.word_embed(discrete_tokens)

                    if args.only_hard_prompt:
                        embeddings = discrete_embeddings
                    elif args.soft_prompt_first:
                        embeddings = torch.cat([ce, discrete_embeddings], dim=1)
                    else:
                        embeddings = torch.cat([discrete_embeddings, ce], dim=1)
                else:
                    embeddings = ce

                # ---- Generate caption ----
                if "gpt" in args.language_model:
                    if args.using_greedy_search:
                        pred = greedy_search(
                            embeddings=embeddings,
                            tokenizer=tokenizer,
                            model=model.gpt
                        )
                    else:
                        pred = beam_search(
                            embeddings=embeddings,
                            tokenizer=tokenizer,
                            model=model.gpt,
                            beam_width=args.beam_width
                        )[0]
                else:
                    pred = opt_search(
                        prompts=args.text_prompt,
                        embeddings=embeddings,
                        tokenizer=tokenizer,
                        model=model.gpt,
                        beam_width=args.beam_width
                    )[0]

                writer.writerow([item["image_id"], pred])

    print(f"[Saved predictions] {args.out_csv}")


# ============================================================
#   STEP 2 — COCOEvalCap with FULL MULTI-GT
# ============================================================
def evaluate_csv(args):
    import json
    from pycocotools.coco import COCO
    from pycocoevalcap.eval import COCOEvalCap

    # ===== LOAD PREDICTIONS =====
    preds = []
    with open(args.out_csv, "r", encoding="utf8") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            img_id = row[0]
            pred = row[1]
            preds.append({"image_id": img_id, "caption": pred})

    # ===== LOAD GT FROM PICKLE =====
    dataset = pickle.load(open(args.pickle, "rb"))

    gt_annotations = []
    gt_images = []
    ann_id = 0

    for item in dataset:
        img_id = item["image_id"]

        # Add to "images" list
        gt_images.append({"id": img_id})

        # Add all GT captions
        for cap in item["captions"]:
            gt_annotations.append({
                "image_id": img_id,
                "id": ann_id,
                "caption": cap
            })
            ann_id += 1

    # ===== SAVE JSON FILES =====
    pred_json = args.out_csv.replace(".csv", "_pred.json")
    gt_json   = args.out_csv.replace(".csv", "_gt.json")

    # Save pred.json (list)
    json.dump(preds, open(pred_json, "w", encoding="utf8"), ensure_ascii=False)

    # Save gt.json (COCO format)
    json.dump({
        "images": gt_images,
        "annotations": gt_annotations
    }, open(gt_json, "w", encoding="utf8"), ensure_ascii=False)

    print("[Evaluating with full GT captions...]")

    # ===== RUN COCO EVAL =====
    coco = COCO(gt_json)
    coco_res = coco.loadRes(pred_json)

    evaluator = COCOEvalCap(coco, coco_res)
    evaluator.evaluate()

    print("\n=== Evaluation Results (COCOEvalCap) ===")
    for k, v in evaluator.eval.items():
        print(f"{k}: {v:.4f}")

# ============================================================
#   MAIN
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--pickle", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--device", default="cuda:0")

    # Model parameters
    parser.add_argument("--language_model", required=True)
    parser.add_argument("--weight_path", required=True)
    parser.add_argument("--continuous_prompt_length", type=int, required=True)
    parser.add_argument("--clip_project_length", type=int, required=True)
    parser.add_argument("--clip_hidden_size", type=int, required=True)
    parser.add_argument("--num_layers", type=int, required=True)

    # Prompt configs
    parser.add_argument("--using_hard_prompt", action="store_true")
    parser.add_argument("--only_hard_prompt", action="store_true")
    parser.add_argument("--soft_prompt_first", action="store_true")

    # Decoding config
    parser.add_argument("--using_greedy_search", action="store_true")
    parser.add_argument("--beam_width", type=int, default=5)

    # Batch size
    parser.add_argument("--batch_size", type=int, default=32)

    args = parser.parse_args()

    generate_predictions(args)
    evaluate_csv(args)
