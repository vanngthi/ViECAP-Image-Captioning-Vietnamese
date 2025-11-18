
echo "Running entity extraction and text features extraction..."
python -m src.data.entity_extractor \
  --image_dir ./dataset/UIT-ViIC/images \
  --caption_path ./dataset/UIT-ViIC/uitviic_captions_test2017.json \
  --out_pickle ./annotations/test_uit_viic_entities.pkl \
  --top_k 5 \
  --mode visual

# python infer_instance.py \
#   --image_path ./dataset/Flick_sportball/images/111796099.jpg \
#   --weight_path ./checkpoints/viecap_vi_multilingual_mask20/vietnamese-0029.pt \
#   --detector_config ./src/config/detector.yaml \
#   --soft_prompt_first \
#   --device cpu
