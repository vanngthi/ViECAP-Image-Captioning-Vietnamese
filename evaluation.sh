#!/bin/bash

echo "Extract features..."

python -m src.data.extract_image_features --model BAAI/AltCLIP-m18 \
                                          --device cuda:0 \
                                          --dataset_path ./dataset/Flick_sportball/test.json \
                                          --img_folder ./dataset/Flick_sportball/images \
                                          --path_of_entities_embeddings ./dataset/vietnamese_entities_embeddings.pickle \
                                          --name_of_entities_text vietnamese_entities \
                                          --path_of_entities ./src/config/vietnamese_entities.json \
                                          --output_path ./annotations/flickr_val_with_features_original.pkl \
                                          --entity_mode original
                                          
python -m src.data.extract_image_features --model BAAI/AltCLIP-m18 \
                                          --device cuda:0 \
                                          --dataset_path ./dataset/Flick_sportball/test.json \
                                          --img_folder ./dataset/Flick_sportball/images \
                                          --detector_config ./src/config/detector.yaml \
                                          --output_path ./annotations/flickr_val_with_features_detect.pkl \
                                          --entity_mode detect
  
WEIGHT_PATH=/DATA/van-n/phenikaa/ViTrCap/checkpoints/viecap_vietnamese_soft_first/viecap_vietnamese_soft_first_latest.pt
NUM_LAYERS=2

python ./evaluation.py \
    --pickle ./annotations/flickr_val_with_features_detect.pkl \
    --out_csv ./annotations/flickr_detect.csv  \
    --device cuda:0 \
    --language_model NlpHUST/gpt2-vietnamese \
    --weight_path $WEIGHT_PATH \
    --continuous_prompt_length 10 \
    --clip_project_length 10 \
    --clip_hidden_size 1024 \
    --num_layers $NUM_LAYERS \
    --using_hard_prompt \
    --soft_prompt_first \
    --beam_width 5 \
    --batch_size 516



python ./evaluation.py \
    --pickle ./annotations/flickr_val_with_features_original.pkl \
    --out_csv ./annotations/flickr_original.csv \
    --device cuda:0 \
    --language_model NlpHUST/gpt2-vietnamese \
    --weight_path $WEIGHT_PATH \
    --continuous_prompt_length 10 \
    --clip_project_length 10 \
    --clip_hidden_size 1024 \
    --num_layers $NUM_LAYERS \
    --using_hard_prompt \
    --soft_prompt_first \
    --beam_width 5 \
    --batch_size 516