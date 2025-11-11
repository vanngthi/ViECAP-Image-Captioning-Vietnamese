import os
import json
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

def download_image(entry, save_dir):
    file_name = entry["file_name"]
    urls = [entry.get("coco_url"), entry.get("flickr_url")]

    dest = os.path.join(save_dir, file_name)
    if os.path.exists(dest):
        return f"Skip (exists): {file_name}"

    for url in urls:
        if not url:
            continue
        try:
            r = requests.get(url, timeout=15)
            if r.status_code == 200:
                with open(dest, "wb") as f:
                    f.write(r.content)
                return f"Downloaded from {url.split('/')[2]}: {file_name}"
        except Exception as e:
            continue

    return f"Failed both links: {file_name}"

if __name__ == "__main__":
    folder = "./UIT-ViIC"
    output_dir = os.path.join(folder, "images")
    os.makedirs(output_dir, exist_ok=True)

    for split in ["train", "val", "test"]:
        print(f"Downloading split: {split}")
        file_path = os.path.join(folder, f"uitviic_captions_{split}2017.json")

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        images = data["images"]

        with ThreadPoolExecutor(max_workers=16) as executor:
            futures = {executor.submit(download_image, img, output_dir): img for img in images}
            for future in tqdm(as_completed(futures), total=len(futures)):
                msg = future.result()
                if msg.startswith("Error") or msg.startswith("HTTP"):
                    print(msg)

        print(f"Finished split {split}. Total: {len(images)} images.")