import csv
import json
import os
from PIL import Image

if __name__ == "__main__":
    folder = "./Flick_sportball"
    img_dir = os.path.join(folder, "images")

    for name in ["train.csv", "test.csv"]:
        csv_path = os.path.join(folder, name)
        out_path = os.path.join(folder, f"{name[:-4]}.json")

        images = {}
        annotations = []

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_name = row["image"].strip()
                img_id = int(row["id"])
                caption = row["caption"].strip()

                img_path = os.path.join(img_dir, img_name)
                if not os.path.exists(img_path):
                    print(f"Missing image: {img_name}")
                    continue

                # Nếu ảnh chưa có thì thêm vào danh sách images
                if img_id not in images:
                    try:
                        with Image.open(img_path) as img:
                            width, height = img.size
                    except Exception:
                        width, height = 0, 0

                    images[img_id] = {
                        "license": 0,
                        "file_name": img_name,
                        "coco_url": "",
                        "height": height,
                        "width": width,
                        "date_captured": "",
                        "flickr_url": "",
                        "id": img_id
                    }

                # Thêm caption vào annotations
                annotations.append({
                    "image_id": img_id,
                    "caption": caption
                })

        coco_data = {
            "images": list(images.values()),
            "annotations": annotations
        }

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(coco_data, f, ensure_ascii=False, indent=2)

        print(f"Saved {out_path}")
        print(f"{len(images)} unique images, {len(annotations)} captions\n")