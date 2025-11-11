import os
import json
import torch
import argparse
import torch.nn.functional as F
from PIL import Image
from transformers import AutoTokenizer, AutoProcessor, AutoModel
from src.detector.detection import ObjectDetector
from src.models.ClipCap import ClipCaptionModel
from src.data.utils import compose_discrete_prompts
from src.data.search import beam_search


@torch.no_grad()
def main(args):
    device = args.device
    clip_hidden_size = 1024  # AltCLIP-m18 → 768 dim, ViECap dùng 1024
    tokenizer = AutoTokenizer.from_pretrained(args.language_model)

    # ===== Load caption model =====
    print("Loading caption model ...")
    model = ClipCaptionModel(
        args.continuous_prompt_length,
        args.clip_project_length,
        clip_hidden_size,
        gpt_type=args.language_model,
    )
    model.load_state_dict(torch.load(args.weight_path, map_location=device), strict=False)
    model.to(device).eval()

    # ===== Load CLIP encoder (AltCLIP) =====
    print("Loading CLIP encoder ...")
    processor = AutoProcessor.from_pretrained(args.clip_model)
    clip_model = AutoModel.from_pretrained(args.clip_model).to(device).eval()

    # ===== Load detector (YOLOv8/11) =====
    print("Loading YOLO object detector ...")
    detector = ObjectDetector(config_path=args.detector_config)

    # ===== Detect objects =====
    print("Detecting objects in image ...")
    detected_entities = list(detector.detect(args.image_path))  # returns set of Vietnamese labels
    print(f"Auto-detected entities ({len(detected_entities)}): {detected_entities}")

    # ===== Encode image features =====
    image = Image.open(args.image_path).convert("RGB")
    image_inputs = processor(images=image, return_tensors="pt").to(device)
    image_features = clip_model.get_image_features(**image_inputs)
    image_features = F.normalize(image_features, dim=-1)
    continuous_embeddings = model.mapping_network(image_features).view(
        -1, args.continuous_prompt_length, model.gpt_hidden_size
    )

    # ===== Create hard prompts =====
    discrete_tokens = compose_discrete_prompts(tokenizer, detected_entities).unsqueeze(0).to(device)
    discrete_embeddings = model.word_embed(discrete_tokens)

    # ===== Combine hard + soft prompts =====
    if args.only_hard_prompt:
        embeddings = discrete_embeddings
    elif args.soft_prompt_first:
        embeddings = torch.cat((continuous_embeddings, discrete_embeddings), dim=1)
    else:
        embeddings = torch.cat((discrete_embeddings, continuous_embeddings), dim=1)

    # ===== Generate caption using beam search =====
    print("Generating caption ...")
    sentence = beam_search(
        embeddings=embeddings,
        tokenizer=tokenizer,
        beam_width=args.beam_width,
        model=model.gpt,
    )[0]

    print(f"\nImage: {os.path.basename(args.image_path)}")
    print(f"Caption: {sentence}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--clip_model", default="BAAI/AltCLIP-m18")
    parser.add_argument("--language_model", default="NlpHUST/gpt2-vietnamese")
    parser.add_argument("--continuous_prompt_length", type=int, default=10)
    parser.add_argument("--clip_project_length", type=int, default=10)
    parser.add_argument("--weight_path", default="./checkpoints/viecap_vietnamese/vietnamese-0034.pt")
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--detector_config", default="./src/config/detector.yaml")  # ✅ Thêm cấu hình detector
    parser.add_argument("--soft_prompt_first", action="store_true", default=True)
    parser.add_argument("--only_hard_prompt", action="store_true", default=False)
    parser.add_argument("--beam_width", type=int, default=5)
    args = parser.parse_args()

    print("Args:", vars(args))
    main(args)
