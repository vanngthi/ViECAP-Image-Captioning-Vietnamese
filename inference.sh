
echo "Running entity extraction and text features extraction..."
# python -m src.data.entity_extractor \
#   --image_dir ./dataset/UIT-ViIC/images \
#   --caption_path ./dataset/UIT-ViIC/uitviic_captions_test2017.json \
#   --out_pickle ./annotations/test_uit_viic_entities.pkl \
#   --top_k 5 \
#   --mode visual

python inference.py \
  --image_path ./images.jpg \
  --weight_path /DATA/van-n/phenikaa/ViTrCap/checkpoints/viecap_vietnamese_soft_first/viecap_vietnamese_soft_first_latest.pt \
  --detector_config ./src/config/detector.yaml \
  --clip_model BAAI/AltCLIP-m18 \
  --language_model NlpHUST/gpt2-vietnamese \
  --using_hard_prompt \
  --continuous_prompt_length 10 \
  --clip_project_length 10 \
  --num_layers 2 \
  --beam_width 5 \
  --device cuda:0 \
  --mode detect
  
  python inference.py \
  --image_path  ./images.jpg \
  --weight_path /DATA/van-n/phenikaa/ViTrCap/checkpoints/viecap_vietnamese_soft_first/viecap_vietnamese_soft_first_latest.pt \
  --clip_model BAAI/AltCLIP-m18 \
  --language_model NlpHUST/gpt2-vietnamese \
  --path_of_entities ./src/config/vietnamese_entities.json \
  --path_of_entities_embeddings ./dataset/vietnamese_entities_embeddings.pickle \
  --name_of_entities_text vietnamese_entities \
  --using_hard_prompt \
  --continuous_prompt_length 10 \
  --clip_project_length 10 \
  --num_layers 2 \
  --top_k 3 \
  --device cuda:0 \

