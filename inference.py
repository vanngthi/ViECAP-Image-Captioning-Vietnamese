import clip
import torch
import argparse
from PIL import Image
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoProcessor

from src.detector.detection import ObjectDetector
from src.models.ClipCap import ClipCaptionModel
from src.data.utils import compose_discrete_prompts
from src.data.load_annotations import load_entities_text
from src.data.search import greedy_search, beam_search, opt_search
from src.data.retrieval_categories import clip_texts_embeddings, image_text_simiarlity, top_k_categories

@torch.no_grad()
def main(args) -> None:
    device = args.device # set device
    
    # load_model
    print("[Start] Loading models...")
    tokenizer = AutoTokenizer.from_pretrained(args.language_model) # tokenizer llm
    model = ClipCaptionModel( # captioning model
        continuous_length=args.continuous_prompt_length,
        clip_project_length=args.clip_project_length,
        clip_hidden_size=args.clip_hidden_size,
        gpt_type=args.language_model,
        num_layers=args.num_layers,
        soft_prompt_first=args.soft_prompt_first,
        only_hard_prompt=args.only_hard_prompt,
    )
    model.load_state_dict(torch.load(args.weight_path, map_location = device), strict = False)
    model.to(device)
    
    processor = AutoProcessor.from_pretrained(args.clip_model) # clip processor
    encoder = AutoModel.from_pretrained(args.clip_model).to(device).eval() # clip encoder
    
    # encode image
    print("[Start] Encoding image...")
    image = Image.open(args.image_path)
    inputs = processor(images=image, return_tensors="pt").to(device)
    
    image_features = encoder.get_image_features(**inputs) 
    image_features = F.normalize(image_features, dim=-1)
    
    # prepare embeddings and hard prompt
    print("Start] Preparing embeddings...")
    continuous_embeddings = model.mapping_network(image_features).view(-1, args.continuous_prompt_length, model.gpt_hidden_size)
    
    if args.mode == "original":
        entities_text = load_entities_text(
            args.name_of_entities_text,
            args.path_of_entities,
            not args.disable_all_entities
        )

        if len(entities_text) == 0:
            print("Failed to load entity vocabulary!")
            return

        # load embedding của vocab (đã encode bằng AltCLIP hoặc CLIP)
        texts_embeddings = clip_texts_embeddings(
            entities_text,
            args.path_of_entities_embeddings
        )
    elif args.mode == "detect":
        detector = ObjectDetector(
            config_path=args.detector_config,
            device=device
        )
        
    
    if args.using_hard_prompt:
        if args.mode == "original":
            logits = image_text_simiarlity(texts_embeddings, temperature = args.temperature, images_features = image_features)
            detected_objects, _ = top_k_categories(entities_text, logits, args.top_k, args.threshold) # List[List[]], [[category1, category2, ...], [], ...]
            detected_objects = detected_objects[0] # infering single image -> List[category1, category2, ...]
        elif args.mode == "detect":
            detected_objects = list(detector.detect(image))
            
        print("Objects in Image:", detected_objects)
        discrete_tokens = compose_discrete_prompts(tokenizer, detected_objects).unsqueeze(dim = 0).to(args.device)

        discrete_embeddings = model.word_embed(discrete_tokens)
        if args.only_hard_prompt:
            embeddings = discrete_embeddings
        elif args.soft_prompt_first:
            embeddings = torch.cat((continuous_embeddings, discrete_embeddings), dim = 1)
        else:
            embeddings = torch.cat((discrete_embeddings, continuous_embeddings), dim = 1)
    else:
        embeddings = continuous_embeddings
        
    # generate caption
    print("[Start] Generating caption...")
    if 'gpt' in args.language_model:
        if not args.using_greedy_search:
            sentence = beam_search(embeddings = embeddings, tokenizer = tokenizer, beam_width = args.beam_width, model = model.gpt) # List[str]
            sentence = sentence[0] # selected top 1
        else:
            sentence = greedy_search(embeddings = embeddings, tokenizer = tokenizer, model = model.gpt)
    else:
        sentence = opt_search(prompts=args.text_prompt, embeddings = embeddings, tokenizer = tokenizer, beam_width = args.beam_width, model = model.gpt)
        sentence=sentence[0]
    
    print(f'[CAPTION]: {sentence}')
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default = 'cuda:0')
    parser.add_argument('--clip_model', default = 'ViT-B/32')
    parser.add_argument('--language_model', default = 'gpt2')
    parser.add_argument('--continuous_prompt_length', type = int, default = 10)
    parser.add_argument('--clip_project_length', type = int, default = 10)
    parser.add_argument("--clip_hidden_size", type=int, default=1024)
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument('--temperature', type = float, default = 0.01)
    parser.add_argument('--top_k', type = int, default = 3)
    parser.add_argument('--threshold', type = float, default = 0.2)
    parser.add_argument('--disable_all_entities', action = 'store_true', default = False, help = 'whether to use entities with a single word only')
    parser.add_argument('--path_of_entities', type=str, default="./src/config/vietnamese_entities.json")
    parser.add_argument('--path_of_entities_embeddings', type=str, default="./dataset/vietnamese_entities_embeddings.pickle")
    parser.add_argument('--name_of_entities_text', default='vietnamese_entities')
    parser.add_argument('--detector_config', type=str, default="./src/config/detector.yaml")
    parser.add_argument('--weight_path', default = './checkpoints/train_coco/coco_prefix-0014.pt')
    parser.add_argument('--image_path', default = './images/')
    parser.add_argument('--using_hard_prompt', action = 'store_true', default = False)
    parser.add_argument('--soft_prompt_first', action = 'store_true', default = False)
    parser.add_argument('--only_hard_prompt', action = 'store_true', default = False)
    parser.add_argument('--using_greedy_search', action = 'store_true', default = False, help = 'greedy search or beam search')
    parser.add_argument('--beam_width', type = int, default = 5, help = 'width of beam')
    parser.add_argument('--text_prompt', type = str, default = None)
    parser.add_argument('--mode', type = str, default = 'original')
    args = parser.parse_args()
    print('args: {}\n'.format(vars(args)))

    main(args)