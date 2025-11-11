
echo "Running entity extraction and text features extraction..."
# python -m src.data.entity_extractor \
#   --image_dir dataset/UIT-ViIC/images \
#   --caption_json dataset/UIT-ViIC/uitviic_captions_train2017.json \
#   --out_pickle annotations/uit_viic_entities.pkl \
#   --top_k 5

python -m src.data.texts_features_extraction \
  --inpath annotations/uit_viic_entities.pkl \
  --outpath annotations/uit_viic_entities_with_features.pkl