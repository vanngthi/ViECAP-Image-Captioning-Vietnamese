import os, json
import torch
import pickle
import argparse
from tqdm import tqdm
import torch.nn.functional as F
from PIL import Image
from pycocotools.coco import COCO
from transformers import AutoProcessor, AutoModel
from collections import defaultdict
from src.data.utils import compose_discrete_prompts
from src.detector.detection import ObjectDetector
from src.data.load_annotations import load_entities_text
from src.data.retrieval_categories import clip_texts_embeddings, image_text_simiarlity, top_k_categories

@torch.no_grad()
def extract_features(encoder, processor, entity_mode, images, caption_map, img_folder, output_path, 
                     path_of_entities, path_of_entities_embeddings, name_of_entities_text, detector_config,
                     temperature, top_k, threshold, device):
    results = []
    
    for item in tqdm(images):
        image_name = item["file_name"]
        image_id = item["id"]
        image_path = os.path.join(img_folder, image_name)

        # Load + encode image
        image = Image.open(image_path)
        inputs = processor(images=image, return_tensors="pt").to(device)
        image_features = encoder.get_image_features(**inputs) 
        image_features = F.normalize(image_features, dim=-1)
        # Get entities
        entities = get_entities(image, image_features, 
                                path_of_entities_embeddings, name_of_entities_text, path_of_entities, 
                                detector_config, 
                                entity_mode, temperature, top_k, threshold, device)
        # Get captions
        captions = caption_map[image_id]

        results.append({
            "image_id": image_name,
            "image_path": image_path,
            "image_features": image_features.cpu().numpy().astype("float32"),
            "entities": entities,
            "captions": captions
        })
    # Save pickle
    with open(output_path, "wb") as f:
        pickle.dump(results, f)
    print("[Saved] =>", output_path)


def get_entities(image, image_features, 
                 path_of_entities_embeddings="./dataset/vietnamese_entities_embeddings.pickle",
                 name_of_entities_text="vietnamese_entities",
                 path_of_entities="./src/config/vietnamese_entities.json",
                 detector_config="./src/config/detector.yaml", mode="original",
                 temperature=0.01, top_k=3, threshold=0.2, device="cuda:0"):
    
    if mode == "detect":
        detector = ObjectDetector(
            config_path=detector_config,
            device=device
        )
        return list(detector.detect(image))
    
    elif mode == "original":
        entities_text = load_entities_text(
            name_of_entities_text,
            path_of_entities,
            not False
        )

        if len(entities_text) == 0:
            print("Failed to load entity vocabulary!")
            return

        # load embedding của vocab (đã encode bằng AltCLIP hoặc CLIP)
        texts_embeddings = clip_texts_embeddings(
            entities_text,
            path_of_entities_embeddings
        )
        
        logits = image_text_simiarlity(texts_embeddings, temperature = temperature, images_features = image_features)
        detected_objects, _ = top_k_categories(entities_text, logits, top_k, threshold) # List[List[]], [[category1, category2, ...], [], ...]
        return detected_objects[0] # infering single image -> List[category1, category2, ...]
    else:
        print("Unknown entity extraction mode:", mode)  
    return []


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Extract image features")
    parser.add_argument("--model", type=str, default="BAAI/AltCLIP-m18")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dataset_path", type=str, default="../dataset/UIT-ViIC/uitviic_captions_val2017.json")
    parser.add_argument("--img_folder", type=str, default="../dataset/UIT-ViIC/images")
    parser.add_argument('--temperature', type = float, default = 0.01)
    parser.add_argument('--path_of_entities', type=str, default="./src/config/vietnamese_entities.json")
    parser.add_argument('--path_of_entities_embeddings', type=str, default="./dataset/vietnamese_entities_embeddings.pickle")
    parser.add_argument('--top_k', type = int, default = 3)
    parser.add_argument('--threshold', type = float, default = 0.2)
    parser.add_argument('--name_of_entities_text', default='vietnamese_entities')
    parser.add_argument('--detector_config', type=str, default="./src/config/detector.yaml")
    parser.add_argument("--output_path", type=str, default="../annotations/uit_viic_val_with_features.pickle")
    parser.add_argument("--entity_mode", type=str, default="original")
    args = parser.parse_args()

    processor = AutoProcessor.from_pretrained(args.model)
    encoder = AutoModel.from_pretrained(args.model).to(args.device).eval()

    # Load data
    with open(args.dataset_path, "r") as f:
        data = json.load(f)

    images = data["images"]
    annotations = data["annotations"]

    caption_map = defaultdict(list)
    for ann in annotations:
        caption_map[ann["image_id"]].append(ann["caption"])

    # Extract features
    extract_features(
        encoder=encoder,
        processor=processor,
        entity_mode=args.entity_mode,
        images=images,
        caption_map=caption_map,
        img_folder=args.img_folder,
        output_path=args.output_path,
        path_of_entities=args.path_of_entities,
        path_of_entities_embeddings=args.path_of_entities_embeddings,
        name_of_entities_text=args.name_of_entities_text,
        detector_config=args.detector_config,
        temperature=args.temperature, device=args.device,
        top_k=args.top_k, threshold=args.threshold
    )
