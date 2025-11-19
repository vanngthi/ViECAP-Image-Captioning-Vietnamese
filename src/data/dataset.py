import torch
import pickle
import random
from typing import Tuple
from torch.utils.data import Dataset

from src.data.utils import parse_entities, padding_captions
from src.data.load_annotations import load_entities_text, load_stopwords


class CaptionsDataset(Dataset):

    def __init__(
        self,
        lm,                    # GPT2LanguageModel (src.llm.gpt2_model)
        clip_encoder,          # CLIPEncoder (src.llm.clip_encoder)
        max_num_of_entities=5,
        using_clip_features=False,
        path_of_datasets='./annotations/viic/viic_with_entities.pkl',
        debug=False,
        args=None
    ) -> None:

        self.args = args
        self.lm = lm
        self.clip_encoder = clip_encoder
        self.using_clip_features = using_clip_features

        # tokenizer VN GPT2
        self.tokenizer = lm.tokenizer

        # CLIP processor (thay clip.tokenize)
        self.clip_processor = clip_encoder.processor

        # Load dataset
        with open(path_of_datasets, "rb") as f:
            captions_with_entities = pickle.load(f)

        # Few-shot
        if args.few_shot_ratio < 1.0:
            random.shuffle(captions_with_entities)
            N = int(len(captions_with_entities) * args.few_shot_ratio)
            captions_with_entities = captions_with_entities[:N]

        # Debug
        if debug:
            captions_with_entities = captions_with_entities[:500]

        # Storage
        self.captions = []
        self.detected_entities = []
        self.captions_lm_tokens = []
        captions_lm_lengths = []

        if using_clip_features:
            self.captions_clip_features = []
        else:
            self.captions_clip_tokens = []

        # ---------------------------------------
        # Build dataset entries
        # ---------------------------------------
        for entry in captions_with_entities:

            if using_clip_features:
                ents, caption, clip_feat = entry
                self.captions_clip_features.append(clip_feat)
            else:
                ents, caption = entry

                # fallback → tokenize bằng AltCLIP processor (77 tokens)
                clip_tok = self.clip_processor(
                    text=caption,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=77
                )
                self.captions_clip_tokens.append(
                    clip_tok["input_ids"].squeeze(0)
                )

            self.captions.append(caption)
            self.detected_entities.append(ents[:max_num_of_entities])

            # GPT2 Vietnam LM tokens
            lm_ids = lm.encode(caption)  # unified 1 function
            self.captions_lm_tokens.append(lm_ids)
            captions_lm_lengths.append(len(lm_ids))

        # Compute max length
        lengths = torch.tensor(captions_lm_lengths, dtype=torch.float32)
        self.max_length_per_caption = min(
            int(lengths.mean() + 10 * lengths.std()),
            int(lengths.max())
        )

        # Load vocab + stopwords
        self.stopwords = load_stopwords()
        self.people_vocabs = [
            "người", "đàn ông", "phụ nữ", "cậu bé", "cô bé", "bé trai", "bé gái", "cô gái", "chàng trai", "nam", "nữ", "trẻ em",
            "bố", "mẹ", "ông", "bà", "cha", "anh trai", "chị gái", "bạn bè", "đồng đội", "đồng nghiệp"
            "vận động viên", "cầu thủ", "thanh niên", "đội", "nhóm",
            "người lính", "bác sĩ", "thầy giáo", "học viên", "học sinh", "thủ môn", "trọng tài", "khán giả", "quân nhân", 
            "hậu vệ", "thủ môn", "tiền đạo"
        ]
        self.objects_vocabs = load_entities_text(
            args.name_of_objects_vocabs,
            args.path_of_objects_vocabs,
            all_entities=False
        )

        print(f"[Dataset] Loaded {len(self.captions)} samples. Max LM len = {self.max_length_per_caption}")


    def __len__(self):
        return len(self.captions)


    def pad_tokens(self, item):
        tokens = self.captions_lm_tokens[item]
        padding = self.max_length_per_caption - len(tokens)

        tokens = tokens[:self.max_length_per_caption]
        if padding > 0:
            tokens = torch.cat([tokens, torch.zeros(padding, dtype=torch.int64) - 1])

        mask = tokens.ge(0)
        tokens[~mask] = 0
        return tokens, mask.float()


    def __getitem__(self, item):
        caption_lm_tokens, mask = self.pad_tokens(item)

        if self.using_clip_features:
            captions_clip = self.captions_clip_features[item]
        else:
            captions_clip = self.captions_clip_tokens[item]

        ents = self.detected_entities[item]

        discrete_tokens = None
        if self.args.using_hard_prompt:
            discrete_tokens = parse_entities(
                self.args,
                self.tokenizer,
                [ents],
                self.stopwords,
                self.people_vocabs,
                self.objects_vocabs
            )[0]

        return (
            self.args,
            captions_clip,
            caption_lm_tokens,
            mask,
            discrete_tokens
        )


def collate(batch):
    args = batch[0][0]
    _, captions_clip, gpt_tokens, masks, discrete_tokens = zip(*batch)

    captions_clip = torch.stack(captions_clip)
    gpt_tokens = torch.stack(gpt_tokens)
    masks = torch.stack(masks)

    if args.using_hard_prompt:
        gpt_tokens, tokens_for_loss, masks, hp_len = padding_captions(
            args, gpt_tokens, masks, discrete_tokens
        )
        return captions_clip, gpt_tokens, tokens_for_loss, masks, hp_len

    else:
        gpt_tokens, tokens_for_loss, masks = padding_captions(
            args, gpt_tokens, masks
        )
        return captions_clip, gpt_tokens, tokens_for_loss, masks, None
