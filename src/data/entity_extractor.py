import os
import json
import pickle
import argparse
from tqdm import tqdm
from src.data.load_annotations import load_captions
from src.detector.detection import ObjectDetector


class EntityExtractor:
    """
        Extract visual entities (YOLO) or text entities (Vietnamese NLP)
    """
    
    def __init__(self, detector_cfg="src/config/detector.yaml"):
        self.detector = ObjectDetector(detector_cfg)

    # =====================================
    #  TEXT ENTITY EXTRACTION (single caption)
    # =====================================
    def extract_caption(self, caption: str, vncorenlp_model, phonlp_model):
        segmented = vncorenlp_model.word_segment(caption)[0]
        tokens, pos_tags, ner_tags, deps = phonlp_model.annotate(text=segmented)

        tokens = tokens[0]
        pos_tags = [p[0] for p in pos_tags[0]]
        ner_tags = ner_tags[0]
        deps = deps[0]

        raw_entities = []
        for tok, pos, ner, dep in zip(tokens, pos_tags, ner_tags, deps):
            head_idx, dep_label = dep
            if (
                pos in {"N", "Nc", "Np"} or 
                dep_label in {"nmod", "sub", "dob", "pob"} or
                ner != "O"
            ):
                raw_entities.append(tok.strip().lower())

        merged_entities = []
        temp = []
        for tok, pos in zip(tokens, pos_tags):
            if pos in {"N", "Nc", "Np"}:
                temp.append(tok.lower())
            else:
                if temp:
                    merged_entities.append(" ".join(temp))
                    temp = []
        if temp:
            merged_entities.append(" ".join(temp))

        merged_entities = [m.replace("_", " ").strip() for m in merged_entities]

        filtered_raw = []
        for r in raw_entities:
            r_clean = r.replace("_", " ").strip()
            if not any(r_clean in m for m in merged_entities):
                filtered_raw.append(r_clean)

        final = list(dict.fromkeys(merged_entities + filtered_raw))
        return final

    # =====================================
    #  TEXT ENTITY EXTRACTION (list captions)
    # =====================================
    def extract_captions(self, dataset_name, caption_path, vncorenlp_model, phonlp_model, out_pickle):
        """
        Load captions internally using load_captions(),
        then extract text entities.
        """
        out_dir = os.path.dirname(os.path.abspath(out_pickle))
        os.makedirs(out_dir, exist_ok=True)

        captions_list = load_captions(dataset_name, caption_path)

        results = []
        print(f"[INFO] Extracting text entities from {len(captions_list)} captions...")

        for caption in tqdm(captions_list):
            entities = self.extract_caption(caption, vncorenlp_model, phonlp_model)
            results.append([entities, caption])

        abs_path = os.path.abspath(out_pickle)
        with open(abs_path, "wb") as f:
            pickle.dump(results, f)

        print(f"[Saved] {len(results)} text-entity samples → {abs_path}")
        return results

    # =====================================
    #  VISUAL ENTITY EXTRACTION (single image)
    # =====================================
    def extract_img(self, image_path: str, top_k=None):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        entities = list(self.detector.detect(image_path))
        if top_k:
            entities = entities[:top_k]
        return entities

    # =====================================
    #  VISUAL ENTITY EXTRACTION (image folder)
    # =====================================
    def extract_imgs(self, image_dir, caption_path, out_pickle, top_k=None):
        os.makedirs(os.path.dirname(out_pickle), exist_ok=True)

        with open(caption_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if "annotations" in data and "images" in data:
            imgid_to_file = {x["id"]: x["file_name"] for x in data["images"]}
            annotations = data["annotations"]
        else:
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
                entities = self.extract_img(img_path, top_k=top_k)
                results.append([entities, caption])
            except Exception as e:
                print(f"[ERROR] {img_path}: {e}")

        with open(out_pickle, "wb") as f:
            pickle.dump(results, f)

        print(f"[Saved] {len(results)} visual samples → {out_pickle}")
        return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract text or visual entities")
    parser.add_argument("--mode", choices=["visual", "text"], default="visual")
    parser.add_argument("--image_dir", required=False)
    parser.add_argument("--caption_path", required=True)
    parser.add_argument("--out_pickle", required=True)
    parser.add_argument("--detector_cfg", default="src/config/detector.yaml")
    parser.add_argument("--top_k", type=int, default=None)

    args = parser.parse_args()

    # import os
    # print("Working directory:", os.getcwd())
    # print("Caption path received:", args.caption_path)
    # print("Absolute:", os.path.abspath(args.caption_path))

    if os.path.exists(args.out_pickle):
        with open(args.out_pickle, "rb") as f:
            data = pickle.load(f)
        print(f"[INFO] Found existing pickle ({len(data)} samples).")
        print("Preview:", data[:3])
        exit()

    extractor = EntityExtractor(detector_cfg=args.detector_cfg)

    if args.mode == "text":
        print("Extracting text entities...")
        import phonlp
        import py_vncorenlp
    
        jar_dir = os.path.abspath("./pretrained/nlp_models/vncorenlp")
        pho_dir = os.path.abspath("./pretrained/nlp_models/phonlp")
        args.caption_path = os.path.abspath(args.caption_path)
        args.out_pickle = os.path.abspath(args.out_pickle)
        
        os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-11-openjdk-amd64"
        os.environ["CLASSPATH"] = os.path.join(jar_dir, "VnCoreNLP-1.2.jar")

        vncorenlp_model = py_vncorenlp.VnCoreNLP(
            save_dir=jar_dir,
            annotators=["wseg"],
            max_heap_size='-Xmx2g'
        )

        phonlp_model = phonlp.load(pho_dir)

        extractor.extract_captions(
            dataset_name="uit_vilc",
            caption_path=args.caption_path,
            vncorenlp_model=vncorenlp_model,
            phonlp_model=phonlp_model,
            out_pickle=args.out_pickle
        )

    else:
        print("Extracting visual entities...")
        extractor.extract_imgs(
            image_dir=args.image_dir,
            caption_path=args.caption_path,
            out_pickle=args.out_pickle,
            top_k=args.top_k
        )
