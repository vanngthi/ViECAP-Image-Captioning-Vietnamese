python eval_caption.py \
    --test_image_dir dataset/test/images \
    --coco_annotation_file dataset/test/annotations_coco.json \
    --weight_path checkpoints/latest.pt \
    --beam_width 5