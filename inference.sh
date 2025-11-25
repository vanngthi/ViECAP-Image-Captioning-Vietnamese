
echo "Running entity extraction and text features extraction..."
# python -m src.data.entity_extractor \
#   --image_dir ./dataset/UIT-ViIC/images \
#   --caption_path ./dataset/UIT-ViIC/uitviic_captions_test2017.json \
#   --out_pickle ./annotations/test_uit_viic_entities.pkl \
#   --top_k 5 \
#   --mode visual

python inference.py \
  --image_path /DATA/van-n/phenikaa/ViTrCap/dataset/UIT-ViIC/images/000000067548.jpg \
  --weight_path ./checkpoints/viecap_vietnamese_20/uiit-vietnamses-20_latest.pt \
  --detector_config ./src/config/detector.yaml \
  --clip_model BAAI/AltCLIP-m18 \
  --language_model NlpHUST/gpt2-vietnamese \
  --using_hard_prompt \
  --continuous_prompt_length 20 \
  --clip_project_length 20 \
  --num_layers 10 \
  --device cpu \
  --mode detect
  
  python inference.py \
  --image_path /DATA/van-n/phenikaa/ViTrCap/dataset/UIT-ViIC/images/000000067548.jpg \
  --weight_path ./checkpoints/viecap_vietnamese_20/uiit-vietnamses-20_latest.pt \
  --clip_model BAAI/AltCLIP-m18 \
  --language_model NlpHUST/gpt2-vietnamese \
  --path_of_entities ./src/config/vietnamese_entities.json \
  --path_of_entities_embeddings ./dataset/vietnamese_entities_embeddings.pickle \
  --name_of_entities_text vietnamese_entities \
  --using_hard_prompt \
  --continuous_prompt_length 20 \
  --clip_project_length 20 \
  --num_layers 10 \
  --top_k 3 \
  --device cpu \

