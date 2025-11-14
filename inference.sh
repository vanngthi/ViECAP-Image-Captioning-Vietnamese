python infer_instance.py \
  --image_path ./dataset/Flick_sportball/images/111796099.jpg \
  --weight_path ./checkpoints/viecap_vi_multilingual_mask20/vietnamese-0029.pt \
  --detector_config ./src/config/detector.yaml \
  --soft_prompt_first \
  --device cpu
