import os
import torch
import argparse
import torch.nn.functional as F
from PIL import Image

from src.llm.gpt2_model import GPT2LanguageModel
from src.llm.clip_encoder import CLIPEncoder
from src.models.ClipCap import ClipCaptionModel
from src.data.utils import compose_discrete_prompts
from src.data.search import beam_search
from src.detector.detection import ObjectDetector


@torch.no_grad()
def run_inference(args):
    device = args.device

    # load model GPT2
    print("[1] Loading GPT2 model ...")
    gpt_model = GPT2LanguageModel(args.language_model, device=device)
    tokenizer = gpt_model.tokenizer

    # Load ClipCap model
    print("[2] Loading ClipCaptionModel ...")
    clip_hidden_size = args.clip_hidden_size

    model = ClipCaptionModel(
        continuous_length=args.continuous_prompt_length,
        clip_project_length=args.clip_project_length,
        clip_hidden_size=clip_hidden_size,
        num_layers=args.num_layers,
        gpt_model=gpt_model,
        soft_prompt_first=args.soft_prompt_first,
        only_hard_prompt=args.only_hard_prompt
    )

    state = torch.load(args.weight_path, map_location=device)
    model.load_state_dict(state, strict=False)
    model.to(device).eval()

    # Load CLIP encoder
    print("[3] Loading CLIP encoder ...")
    clip_encoder = CLIPEncoder(args.clip_model, device=device)

    # Load YOLO object detector
    print("[4] Loading YOLO detector ...")
    detector = ObjectDetector(config_path=args.detector_config)


    # Detect objects
    print("[5] Detecting objects in image ...")
    detected_entities = detector.detect(args.image_path)  
    detected_entities = list(detected_entities)
    print("Detected entities:", detected_entities)

    # Encode image using CLIP
    print("[6] Encoding image features ...")
    image = Image.open(args.image_path).convert("RGB")
    img_features = clip_encoder.encode_image(image)  # (1, hidden)
    img_features = F.normalize(img_features, dim=-1)  # (1, hidden)

    # Convert CLIP vector → GPT2 prefix embeddings
    clip_prefix = model.mapping_network(img_features)
    clip_prefix = clip_prefix.view(1, args.continuous_prompt_length, model.gpt_hidden_size)

    # 7) Build hard prompt embeddings
    print("[7] Building hard prompt tokens ...")
    if len(detected_entities) > 0:
        discrete_ids = compose_discrete_prompts(tokenizer, detected_entities)
        discrete_ids = discrete_ids.unsqueeze(0).to(device)     # (1, N)
        hard_embeddings = model.word_embed(discrete_ids)        # (1, N, hidden)
    else:
        hard_embeddings = None

    # 8) Combine prefix + hard prompt
    print("[8] Combining prompts ...")

    if hard_embeddings is None:
        embeddings = clip_prefix

    elif args.only_hard_prompt:
        embeddings = hard_embeddings

    elif args.soft_prompt_first:
        embeddings = torch.cat((clip_prefix, hard_embeddings), dim=1)

    else:
        embeddings = torch.cat((hard_embeddings, clip_prefix), dim=1)

    # 9) Beam Search GENERATION
    print("[9] Running beam search ...")
    caption = beam_search(
        embeddings=embeddings,
        tokenizer=tokenizer,
        model=model.gpt,
        beam_width=args.beam_width,
    )[0]

    print("\n========== RESULT ==========")
    print("Image:", os.path.basename(args.image_path))
    print("Caption:", caption)
    print("============================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--image_path", required=True)
    parser.add_argument("--clip_model", default="BAAI/AltCLIP-m18")
    parser.add_argument("--language_model", default="NlpHUST/gpt2-vietnamese")
    parser.add_argument("--continuous_prompt_length", type=int, default=10)
    parser.add_argument("--clip_project_length", type=int, default=10)
    parser.add_argument("--clip_hidden_size", type=int, default=1024)
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--weight_path", type=str, required=True)
    parser.add_argument("--detector_config", default="./src/config/detector.yaml")
    parser.add_argument("--soft_prompt_first", action="store_true", default=True)
    parser.add_argument("--only_hard_prompt", action="store_true", default=False)
    parser.add_argument("--beam_width", type=int, default=5)

    args = parser.parse_args()
    print(args)
    run_inference(args)
