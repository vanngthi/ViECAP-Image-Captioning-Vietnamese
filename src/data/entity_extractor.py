import os
import json
import pickle
import argparse
from tqdm import tqdm
from src.detector.detection import ObjectDetector


class EntityExtractor:
    """
        Use ObjectDetector to extract visual entities from images
    """
    
    def __init__(self, detector_cfg="src/config/detector.yaml"):
        self.detector = ObjectDetector(detector_cfg) # create detector instance

    # extract single image
    def extract(self, image_path: str, top_k=None) -> list:
        """
            Extract visual entities from a single image
            args:
                image_path: str
                top_k: int or None, limit number of entities
            return:
                list of entitiesl
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        entities = list(self.detector.detect(image_path))
        if top_k:
            entities = entities[:top_k]
        return entities

    # extract entities from a folder of images
    def extract_folder(self, image_dir, caption_path, out_pickle, top_k=None):
        """
            Extract visual entities from a folder of images and save to pickle
            args:
                image_dir: str, folder containing images
                caption_path: str, path to caption json file
                out_pickle: str, path to output pickle file
                top_k: int or None, limit number of entities per image
            return:
                list of [entities, caption]
        """
        out_dir = os.path.dirname(out_pickle)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
            print(f"Created directory: {out_dir}")

        # if pickle exists, load and return
        if os.path.exists(out_pickle):
            print(f"Found existing pickle: {out_pickle}")
            with open(out_pickle, "rb") as f:
                data = pickle.load(f)
            print(f"Loaded {len(data)} samples.")
            print("Preview:", data[:3])
            return data

        # load captions
        with open(caption_json, "r", encoding="utf-8") as f:
            data = json.load(f)

        # support COCO format
        if "annotations" in data and "images" in data:
            imgid_to_file = {x["id"]: x["file_name"] for x in data["images"]}
            annotations = data["annotations"]
        else:
            # Simple format
            annotations = [{"image_id": i, "caption": x["caption"], "file_name": x["image"]}
                           for i, x in enumerate(data)]
            imgid_to_file = {x["image_id"]: x["file_name"] for x in annotations}

        results = []

        print("Extracting visual entities...")
        for ann in tqdm(annotations):
            img_path = os.path.join(image_dir, imgid_to_file[ann["image_id"]])
            caption = ann["caption"]

            if not os.path.exists(img_path):
                print(f"Missing: {img_path}")
                continue

            try:
                entities = self.extract_image(img_path, top_k=top_k)
                results.append([entities, caption])
            except Exception as e:
                print(f"Error {img_path}: {e}")

        # Save pickle
        with open(out_pickle, "wb") as f:
            pickle.dump(results, f)

        print(f"Saved {len(results)} samples → {out_pickle}")
        print("Preview:", results[:3])

        return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract visual entities from images using detector")
    parser.add_argument("--image_dir", required=True, help="Thư mục chứa ảnh (ví dụ: dataset/Flick_sportball/image)")
    parser.add_argument("--caption_path", required=True, help="File JSON chứa captions")
    parser.add_argument("--out_pickle", required=True, help="Đường dẫn file pickle đầu ra")
    parser.add_argument("--detector_cfg", default="src/config/detector.yaml", help="Cấu hình detector YOLO")
    parser.add_argument("--top_k", type=int, default=None, help="Giới hạn số entity tối đa mỗi ảnh")

    args = parser.parse_args()

    extractor = EntityExtractor(detector_cfg=args.detector_cfg)
    extractor.extract_folder(
        image_dir=args.image_dir,
        caption_path=args.caption_path,
        out_pickle=args.out_pickle,
        top_k=args.top_k
    )
