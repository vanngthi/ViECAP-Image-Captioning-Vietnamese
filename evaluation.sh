#!/bin/bash

# echo "Extract features..."

# python -m src.data.extract_image_features --model BAAI/AltCLIP-m18 \
#                                           --device cpu \
#                                           --dataset_path ./dataset/UIT-ViIC/uitviic_captions_val2017.json \
#                                           --img_folder ./dataset/UIT-ViIC/images \
#                                           --path_of_entities_embeddings ./dataset/vietnamese_entities_embeddings.pickle \
#                                           --name_of_entities_text vietnamese_entities \
#                                           --path_of_entities ./src/config/vietnamese_entities.json \
#                                           --output_path ./annotations/uit_viic_val_with_features_original.pkl \
#                                           --entity_mode original
                                          
# python -m src.data.extract_image_features --model BAAI/AltCLIP-m18 \
#                                           --device cpu \
#                                           --dataset_path ./dataset/UIT-ViIC/uitviic_captions_val2017.json \
#                                           --img_folder ./dataset/UIT-ViIC/images \
#                                           --detector_config ./src/config/detector.yaml \
#                                           --output_path ./annotations/uit_viic_val_with_features_detect.pkl \
#                                           --entity_mode detect
  


python ./evaluation.py \
    --pickle ./annotations/uit_viic_val_with_features_detect.pkl \
    --out_csv ./annotations/uit_viic_detect.csv  \
    --device cuda:0 \
    --language_model NlpHUST/gpt2-vietnamese \
    --weight_path ./checkpoints/viecap_vietnamese_20/uiit-vietnamses-20_latest.pt \
    --continuous_prompt_length 20 \
    --clip_project_length 20 \
    --clip_hidden_size 1024 \
    --num_layers 10 \
    --using_hard_prompt \
    --beam_width 5 \
    --batch_size 516



python ./evaluation.py \
    --pickle ./annotations/uit_viic_val_with_features_original.pkl \
    --out_csv ./annotations/uit_viic_original.csv \
    --device cuda:0 \
    --language_model NlpHUST/gpt2-vietnamese \
    --weight_path ./checkpoints/viecap_vietnamese_20/uiit-vietnamses-20_latest.pt \
    --continuous_prompt_length 20 \
    --clip_project_length 20 \
    --clip_hidden_size 1024 \
    --num_layers 10 \
    --using_hard_prompt \
    --beam_width 5 \
    --batch_size 516