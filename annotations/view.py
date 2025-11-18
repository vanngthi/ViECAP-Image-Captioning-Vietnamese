import os
import pickle

def export_captions_txt(pkl_file, save_path):
    """
    Export both entities and captions from the PKL file into a readable .txt
    Format:
        Caption: ...
        Entities: ...
        -----
    """
    # Load PKL file (format: [[entities, caption], ...])
    with open(pkl_file, "rb") as f:
        data = pickle.load(f)

    # Ensure directory exists
    out_dir = os.path.dirname(os.path.abspath(save_path))
    os.makedirs(out_dir, exist_ok=True)

    with open(save_path, "w", encoding="utf-8") as f:
        for entities, caption in data:
            f.write(f"Caption: {caption}\n")
            f.write("Entities: " + ", ".join(entities) + "\n")
            f.write("-" * 50 + "\n")

    print(f"[Saved] Exported {len(data)} items → {save_path}")


export_captions_txt("./uit_viic_entities.pkl", "./uit_viic_entities.txt")