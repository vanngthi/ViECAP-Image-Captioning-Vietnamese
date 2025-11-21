import os
import wandb
import sys
import clip
import torch
import random
import argparse
import numpy as np
from tqdm import tqdm
import torch.nn.functional as nnf
from src.data.utils import noise_injection
from torch.utils.data import DataLoader
from src.data.dataset import CaptionsDataset, collate
from src.llm.clip_encoder import CLIPEncoder
from src.llm.gpt2_model import GPT2LanguageModel
from src.models.ClipCap import ClipCaptionModel, ClipCaptionPrefix
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from transformers import get_cosine_schedule_with_warmup
from transformers import AutoModel, AutoProcessor

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train(
    args,
    datasets: CaptionsDataset,
    model: ClipCaptionModel,
    warmup_steps: int = 5000,
    output_dir: str = '.',
    output_prefix: str = ''
):
    device = args.device
    batch_size = args.bs
    epochs = args.epochs

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model = model.to(device)
    model.train()
    if not args.using_clip_features:
        processor = AutoProcessor.from_pretrained(args.clip_model)
        encoder = AutoModel.from_pretrained(args.clip_model).to(device)
        encoder.eval()

    optimizer = AdamW(model.parameters(), lr=args.lr)
    dataloader = DataLoader(
        datasets,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        collate_fn=collate
    )

    tokenizer = dataloader.dataset.tokenizer
    total_steps = epochs * len(dataloader)
    warmup_steps = int(0.05 * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=epochs * len(dataloader)
    )

    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)

    for epoch in range(epochs):
        sys.stdout.flush()
        print(f">>> Training epoch {epoch}")
        progress = tqdm(total=len(dataloader), desc=output_prefix)

        train_loss_sum = 0
        epoch_loss_accum = 0

        for idx, (captions_clip,
                  captions_gpt_tokens,
                  captions_tokens_for_loss,
                  masks,
                  hard_prompts_length) in enumerate(dataloader):

            model.zero_grad()

            # -------- prefix extraction --------
            if not args.using_clip_features:
                with torch.no_grad():
                    text_inputs = processor(
                        text=[datasets.captions[i] for i in range(batch_size)],
                        return_tensors="pt",
                        padding=True
                    ).to(device)
                    continuous_prefix = encoder.get_text_features(**text_inputs).float()
            else:
                continuous_prefix = captions_clip.to(device).float()

            if args.normalize_prefix:
                continuous_prefix /= continuous_prefix.norm(2, dim=-1, keepdim=True)

            continuous_prefix = noise_injection(
                continuous_prefix,
                variance=args.noise_variance,
                device=device
            )

            captions_gpt_tokens = captions_gpt_tokens.to(device)
            captions_tokens_for_loss = captions_tokens_for_loss.to(device)
            masks = masks.to(device)

            # -------- forward pass --------
            with torch.cuda.amp.autocast(enabled=args.use_amp):
                if args.using_hard_prompt:
                    outputs = model(
                        continuous_prefix,
                        captions_gpt_tokens,
                        hard_prompts_length,
                        masks
                    )
                    logits = outputs.logits
                else:
                    outputs = model(
                        continuous_prefix,
                        captions_gpt_tokens,
                        mask=masks
                    )
                    logits = outputs.logits

            captions_tokens_for_loss = captions_tokens_for_loss.masked_fill(
                captions_tokens_for_loss == tokenizer.eos_token_id, 0
            )

            loss = nnf.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                captions_tokens_for_loss.flatten(),
                ignore_index=0
            )

            # -------- backward + optimizer --------
            scaler.scale(loss).backward()

            # -------- gradient norm logging --------
            grad_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    grad_norm += (p.grad.data.norm(2).item())**2
            grad_norm = grad_norm**0.5
            wandb.log({"train/grad_norm": grad_norm})

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

            # -------- basic logs --------
            progress.set_postfix({"loss": loss.item()})
            progress.update()

            wandb.log({
                "train/loss_iter": loss.item(),
                "train/lr": optimizer.param_groups[0]["lr"]
            })

            epoch_loss_accum += loss.item()
            train_loss_sum += loss.item()

            # print loss block
            log_iters = len(dataloader)//5 if len(dataloader) > 5 else len(dataloader)
            if (idx + 1) % log_iters == 0:
                avg_loss = train_loss_sum / log_iters
                print(f'epoch {epoch}, iter {idx}, avg loss: {avg_loss}')
                wandb.log({"train/loss_avg_block": avg_loss})
                train_loss_sum = 0

                latest_ckpt = os.path.join(output_dir, f"{output_prefix}_latest.pt")
                torch.save(model.state_dict(), latest_ckpt)
                wandb.save(latest_ckpt)

        progress.close()

        # -------- epoch loss log --------
        epoch_avg_loss = epoch_loss_accum / len(dataloader)
        wandb.log({
            "train/loss_epoch": epoch_avg_loss,
            "epoch": epoch
        })
        print(f">>> Epoch {epoch} avg loss: {epoch_avg_loss}")

        # -------- save checkpoint --------
        if (epoch + 1) % args.save_every == 0 or epoch == epochs - 1:
            ckpt_path = os.path.join(output_dir, f"{output_prefix}-00{epoch}.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saving checkpoint → {ckpt_path}")
            wandb.save(ckpt_path)

        model.train()


if __name__ == '__main__':
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    parser = argparse.ArgumentParser()
    parser.add_argument('--bs', type = int, default = 80, help = 'batch size')
    parser.add_argument('--lr', type = float, default = 2e-5, help = 'learning rate for training')
    parser.add_argument('--device', default = 'cuda:0', help = 'gpu for training')
    parser.add_argument('--epochs', type = int, default = 15, help = 'number of epochs')
    parser.add_argument('--random_mask', action = 'store_true', default = False, help = 'entity masking strategy')
    parser.add_argument('--prob_of_random_mask', type = float, default = 0.4, help = 'masking rate')
    parser.add_argument('--clip_project_length', type = int, default = 10, help = 'clip projecting length')
    parser.add_argument('--continuous_prompt_length', type = int, default = 10, help = 'soft prompts length')
    parser.add_argument('--max_num_of_entities', type = int, default = 10, help = 'maximum number of detected entities')
    parser.add_argument('--prompt_template_length', type = int, default = 5, help = 'maximum number of hard prompt entities')
    parser.add_argument('--num_layers', type = int, default = 8, help = 'number of layer in Transformer-based projector')
    parser.add_argument('--noise_variance', type = float, default = 0.016, help = 'noise variance')
    parser.add_argument('--clip_model', default = "BAAI/AltCLIP-m18", help = "'RN50', 'RN101', 'RN50x4', 'ViT-B/32'")
    parser.add_argument('--using_clip_features', action = 'store_true', default = False, help = 'whether to use the pre-extracted features')
    parser.add_argument('--is_rn', dest = 'is_rn', action = 'store_true', default = False, help = 'CLIP backbone: True -> ResNet, False -> ViT')
    parser.add_argument('--language_model', default = 'NlpHUST/gpt2-vietnamese', help = 'gpt2, facebook/opt-350m')
    parser.add_argument('--using_hard_prompt', action = 'store_true', default = False, help = 'whether to entity-aware hard prompts')
    parser.add_argument('--soft_prompt_first', action = 'store_true', default = False, help = 'True -> soft prompt first, i.e., soft prompt + hard prompt')
    parser.add_argument('--only_hard_prompt', action = 'store_true', default = False, help = 'True -> do not use soft prompts in this case')
    parser.add_argument('--debug', action = 'store_true', default = False, help = 'debug = True means using a smaller dataloader')
    parser.add_argument('--few_shot_ratio', type = float, default = 1.0, help = 'measuring the low-data setting')
    parser.add_argument('--save_every', type = int, default = 1, help = 'save weights every n epochs')
    parser.add_argument('--prefix', default = 'coco_prefix', help = 'prefix name for saved weights')
    parser.add_argument('--path_of_datasets', default = './annotations/coco/coco_with_entities.pickle')
    parser.add_argument('--out_dir', default = './checkpoints', help = 'the path of output')
    parser.add_argument('--normalize_prefix', dest = 'normalize_prefix', type = int, default = True, help = 'normalizing prefix')
    parser.add_argument('--name_of_objects_vocabs', default = 'visual_genome_entities')
    parser.add_argument('--path_of_objects_vocabs', default = './annotations/vocabulary/all_objects_attributes_relationships.pickle')
    parser.add_argument('--frozen_gpt', action = 'store_true', default = False, help = 'freezing language models during training')
    parser.add_argument('--num_workers', type = int, default = 0)
    parser.add_argument('--use_amp', action = 'store_true', default = False, help = "whether to use torch.amp to acclerate training")
    parser.add_argument('--disable_random_seed', action = 'store_true', default = False, help = 'set random seed for reproducing')
    parser.add_argument('--random_seed', type = int, default = 30, help = 'set random seed for reproducing')
    
    args = parser.parse_args()
    print(f'args: {vars(args)}')
    if not args.disable_random_seed:
        set_seed(args.random_seed)

    # setting wandb
    run = wandb.init(
        project="ViECAP Model Training",
        name=f"{args.prefix}-lr{args.lr}-bs{args.bs}-ep{args.epochs}",
        config=vars(args)
    )
    
    clip_hidden_size = 1024

    datasets = CaptionsDataset(
        language_model = args.language_model,
        max_num_of_entities = args.max_num_of_entities,
        using_clip_features = args.using_clip_features,
        path_of_datasets = args.path_of_datasets,
        debug = args.debug,
        args = args
    )
    
    sample = datasets[2]
    
    args, captions_clip, captions_gpt_tokens, masks, discrete_tokens = sample

    print("---- Dataset Sample ----")
    print("Entities:", datasets.detected_entities[2])
    print("Caption:", datasets.captions[2])
    print("CLIP tokens shape:", captions_clip.shape)
    print("GPT tokens:", captions_gpt_tokens[:20])
    print("Mask:", masks[:20])
    print("Discrete tokens:", discrete_tokens)
    if discrete_tokens is not None:
        prompt_text = datasets.tokenizer.decode(discrete_tokens.tolist(), skip_special_tokens=True)
        print("Hard prompt (decoded):", prompt_text)
    else:
        print("No hard prompt (using_hard_prompt=False)")
        
    print("------------------------")

    # sample = datasets[4]
    
    # args, captions_clip, captions_gpt_tokens, masks, discrete_tokens = sample

    # print("---- Dataset Sample ----")
    # print("Entities:", datasets.detected_entities[4])
    # print("Caption:", datasets.captions[4])
    # print("CLIP tokens shape:", captions_clip.shape)
    # print("GPT tokens:", captions_gpt_tokens[:20])
    # print("Mask:", masks[:20])
    # print("Discrete tokens:", discrete_tokens)
    # if discrete_tokens is not None:
    #     prompt_text = datasets.tokenizer.decode(discrete_tokens.tolist(), skip_special_tokens=True)
    #     print("Hard prompt (decoded):", prompt_text)
    # else:
    #     print("No hard prompt (using_hard_prompt=False)")
        
    # print("------------------------")
    
    if args.frozen_gpt:
        model = ClipCaptionPrefix(args.continuous_prompt_length, args.clip_project_length, clip_hidden_size, args.num_layers, gpt_type = args.language_model, soft_prompt_first = args.soft_prompt_first, only_hard_prompt = args.only_hard_prompt)
    else:
        model = ClipCaptionModel(args.continuous_prompt_length, args.clip_project_length, clip_hidden_size, args.num_layers, gpt_type = args.language_model, soft_prompt_first = args.soft_prompt_first, only_hard_prompt = args.only_hard_prompt)
    
    train(args, datasets, model, output_dir = args.out_dir, output_prefix = args.prefix)