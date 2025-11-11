
# echo "Running entity extraction and text features extraction..."
# python -m src.data.entity_extractor \
#   --image_dir dataset/UIT-ViIC/images \
#   --caption_json dataset/UIT-ViIC/uitviic_captions_train2017.json \
#   --out_pickle annotations/uit_viic_entities.pkl \
#   --top_k 5

# python -m src.data.texts_features_extraction \
#   --inpath annotations/uit_viic_entities.pkl \
#   --outpath annotations/uit_viic_entities_with_features.pkl

EXP_NAME="viecap_vi_multilingual_mask20"
mkdir -p ./logs/$EXP_NAME
LOG_FILE=./logs/$EXP_NAME/$(date "+%Y-%m-%d-%H-%M-%S").log

echo "Training model ..."
python train.py \
  --using_clip_features \
  --using_hard_prompt \
  --bs 64 \
  --lr 2e-5 \
  --epochs 50 \
  --device cuda:0 \
  --clip_model "BAAI/AltCLIP-m18" \
  --language_model NlpHUST/gpt2-vietnamese \
  --random_mask \
  --prob_of_random_mask 0.2 \
  --out_dir ./checkpoints/$EXP_NAME \
  --path_of_datasets annotations/uit_viic_entities_with_features.pkl \
  --name_of_objects_vocabs vietnamese_entities \
  --path_of_objects_vocabs src/config/vietnamese_entities.json \
  --use_amp \
  |& tee -a ${LOG_FILE}

