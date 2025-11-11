# src/data/entity_extractor.py
import os
import json
import pickle
from tqdm import tqdm
from src.detector.detection import ObjectDetector


class EntityExtractor:
    def __init__(self, detector_cfg="src/config/detector.yaml"):
        self.detector = ObjectDetector(detector_cfg)

    def extract_from_dataset(self, image_dir, caption_json, out_pickle, top_k=None):
        out_dir = os.path.dirname(out_pickle)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
            print(f"Created directory: {out_dir}")
            
        # Nếu file pickle đã tồn tại → đọc lại
        if os.path.exists(out_pickle):
            print(f"Found existing pickle file: {out_pickle}")
            with open(out_pickle, "rb") as f:
                data = pickle.load(f)
            print(f"Loaded {len(data)} samples.")
            print("Preview:")
            for sample in data[:5]:
                print(sample)
            return data

        print(f"Start extracting entities using detector...")
        with open(caption_json, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Hỗ trợ cả format COCO và format đơn giản
        if "annotations" in data and "images" in data:
            imgid_to_file = {x["id"]: x["file_name"] for x in data["images"]}
            annotations = data["annotations"]
        else:
            annotations = [{"image_id": i, "file_name": x["image"], "caption": x["caption"]}
                           for i, x in enumerate(data)]
            imgid_to_file = {x["image_id"]: x["file_name"] for x in annotations}

        results = []

        for ann in tqdm(annotations, desc="Extracting visual entities"):
            img_path = os.path.join(image_dir, imgid_to_file[ann["image_id"]])
            caption = ann["caption"]

            if not os.path.exists(img_path):
                print(f"Missing image: {img_path}")
                continue

            try:
                entities = list(self.detector.detect(img_path))
                if top_k:
                    entities = entities[:top_k]
                results.append([entities, caption])
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

        # Ghi kết quả ra pickle
        with open(out_pickle, "wb") as f:
            pickle.dump(results, f)
        print(f"Saved {len(results)} samples → {out_pickle}")

        # In preview
        print("Preview:")
        for r in results[:5]:
            print(r)

        return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract visual entities from images using detector")
    parser.add_argument("--image_dir", required=True, help="Thư mục chứa ảnh (ví dụ: dataset/Flick_sportball/image)")
    parser.add_argument("--caption_json", required=True, help="File JSON chứa captions")
    parser.add_argument("--out_pickle", required=True, help="Đường dẫn file pickle đầu ra")
    parser.add_argument("--detector_cfg", default="src/config/detector.yaml", help="Cấu hình detector YOLO")
    parser.add_argument("--top_k", type=int, default=None, help="Giới hạn số entity tối đa mỗi ảnh")

    args = parser.parse_args()

    extractor = EntityExtractor(detector_cfg=args.detector_cfg)
    extractor.extract_from_dataset(
        image_dir=args.image_dir,
        caption_json=args.caption_json,
        out_pickle=args.out_pickle,
        top_k=args.top_k
    )
