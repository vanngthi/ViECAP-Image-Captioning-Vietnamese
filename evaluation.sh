#!/bin/bash

# ================================
# CONFIG chung
# ================================
WEIGHT="./checkpoints/viecap_vietnamese/uiit-vietnamses_latest.pt"
CLIP_MODEL="BAAI/AltCLIP-m18"
LANGUAGE_MODEL="NlpHUST/gpt2-vietnamese"
DEVICE="cpu"

IMG_DIR="./dataset/UIT-ViIC/images"
GT_JSON="./dataset/UIT-ViIC/uitviic_captions_val2017.json"
SAVE_DIR="./eval_results"


# ================================
# MODE 1 — DETECTOR MODE
# ================================
echo "==========================================="
echo "   RUNNING EVALUATION — MODE: detect"
echo "==========================================="

python evaluation.py \
  --weight_path $WEIGHT \
  --clip_model $CLIP_MODEL \
  --language_model $LANGUAGE_MODEL \
  --detector_config ./src/config/detector.yaml \
  --using_hard_prompt \
  --continuous_prompt_length 10 \
  --clip_project_length 10 \
  --num_layers 10 \
  --device $DEVICE \
  --mode detect \
  --coco_images_dir $IMG_DIR \
  --coco_gt_json $GT_JSON \
  --save_dir $SAVE_DIR/detect_mode


# ================================
# MODE 2 — ORIGINAL MODE (ENTITIES)
# ================================
echo "==========================================="
echo "   RUNNING EVALUATION — MODE: original"
echo "==========================================="

python evaluation.py \
  --weight_path $WEIGHT \
  --clip_model $CLIP_MODEL \
  --language_model $LANGUAGE_MODEL \
  --path_of_entities ./src/config/vietnamese_entities.json \
  --path_of_entities_embeddings ./dataset/vietnamese_entities_embeddings.pickle \
  --name_of_entities_text vietnamese_entities \
  --using_hard_prompt \
  --continuous_prompt_length 10 \
  --clip_project_length 10 \
  --num_layers 10 \
  --top_k 3 \
  --device $DEVICE \
  --mode original \
  --coco_images_dir $IMG_DIR \
  --coco_gt_json $GT_JSON \
  --save_dir $SAVE_DIR/original_mode
