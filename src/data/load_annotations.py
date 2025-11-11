import os, json
from tqdm import tqdm
from typing import List
from underthesea import text_normalize

def load_viecap_captions(path: str) -> List[str]:
    print("Loading caption ... ...")
    with open(path, 'r', encoding="utf-8") as infile:
        data = json.load(infile)               # dictionary -> {image_path: List[caption1, caption2, ...]}
    punctuations = ['!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', ' ', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~']

    captions = []
    
    if isinstance(data, dict) and "annotations" in data:
        for ann in tqdm(data["annotations"]):
            caption = ann.get("caption", "")
            caption = text_normalize(caption).strip()
            if caption.isupper():                      # processing the special caption in the COCO Caption, e.g., 'A BOY IS PLAYING BASEBALL.'
                caption = caption.lower()
            caption = caption[0].upper() + caption[1:] # capitalizing the first letter in the caption
            if caption[-1] not in punctuations:        # adding a '.' at the end of the caption if there are no punctuations.
                caption += ' .'
            captions.append(caption)
        
    return captions


def load_captions(name_of_dataset: str, path_of_dataset: str) -> List[str]:
    if name_of_dataset in ["uit_vilc", "flick_sportball"]:
        return load_viecap_captions(path_of_dataset)

    print(f"Dataset '{name_of_dataset}' chưa được hỗ trợ.")
    return []

def load_vietnamese_entities(path: str, all_entities: bool = True) -> List[str]:
    with open(path, 'r', encoding='utf-8') as infile:
        entities = json.load(infile)  # Expecting a list like ["người", "xe đạp", "con chó", ...]

    if all_entities:
        entities = [entity.lower().strip() for entity in entities]
    else:
        entities = [entity.lower().strip() for entity in entities if len(entity.split()) == 1]

    entities = list(sorted(set(entities)))  # Remove duplicates and sort
    return entities

def load_entities_text(name_of_entities: str, path_of_entities: str, all_entities: bool = True) -> List[str]:
    if name_of_entities == 'vietnamese_entities':
        return load_vietnamese_entities(path_of_entities, all_entities)
    
    print('The entities text fails to load!')
    return []


def load_stopwords() -> list[str]:
    stopwords = {
        "và", "là", "ở", "có", "cho", "với", "như", "khi", "này", "kia", "ấy", "đó", "được", "bị", "sẽ", "đang",
        "rằng", "thì", "nếu", "vì", "do", "nên", "mà", "thôi", "chứ", "mỗi", "đều", "các", "những", "một", "hai",
        "ba", "bốn", "năm", "nhiều", "ít", "lúc", "nơi", "nào", "đâu", "tại", "trên", "dưới", "trong", "ngoài",
        "giữa", "vào", "ra", "qua", "lại", "nữa", "đến", "tới", "của", "bằng", "vẫn", "đang", "vừa", "mới",
        "cũng", "còn", "chỉ", "thật", "rất", "quá", "đã", "chưa", "nên", "vì", "tuy", "dù", "nhưng", "hay",
        "hoặc", "vậy", "nên", "thế", "do", "từ", "để", "kể", "song", "cả", "đôi", "nhiều", "ít", "đôi khi",
        "từng", "hầu như", "gần như", "luôn", "thường", "suốt", "đều", "ai", "gì", "nào", "bao nhiêu", "sao",
        "ấy", "kia", "đó", "chẳng", "chưa", "không", "chả", "chẳng", "đừng", "hãy", "đi", "nha", "nhé",
        "thôi", "mà", "cơ", "nhỉ", "ờ", "ừ", "ờm"
    }

    punctuations = ['!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', ' ', '-', '.', '/', ':', ';',
                    '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~']

    # Các từ mô tả hình ảnh nên loại bỏ trong prompt
    other_words = {"ảnh", "hình", "hình ảnh", "bức", "tấm", "bức ảnh", "bức hình", "nền", "phía", "cạnh", "bên", "trước", "sau"}

    stopwords_and_punctuations = stopwords.union(punctuations)
    stopwords_and_punctuations = stopwords_and_punctuations.union(other_words)
    stopwords_and_punctuations = [w.lower().strip() for w in stopwords_and_punctuations if w.strip()]
    stopwords_and_punctuations.sort()

    return stopwords_and_punctuations


if __name__ == "__main__":
    dataset_name = "uit_vilc"
    # dataset_path = "../dataset/UIT-ViIC/uitviic_captions_test2017.json"
    dataset_path = "../dataset/Flick_sportball/test.json"

    captions = load_captions(dataset_name, dataset_path)
    print(f"Loaded {len(captions)} captions.")
    print("\nVí dụ 5 caption đầu:")
    for cap in captions[:5]:
        print("-", cap)