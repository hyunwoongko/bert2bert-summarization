import json
import sys

import torch
from tqdm import tqdm

from transformers import EncoderDecoderConfig, BertConfig, EncoderDecoderModel
from bert2bert import KoBertTokenizer


@torch.no_grad()
def inference():
    step = sys.argv[1]
    encoder_config = BertConfig.from_pretrained("monologg/kobert")
    decoder_config = BertConfig.from_pretrained("monologg/kobert")
    config = EncoderDecoderConfig.from_encoder_decoder_configs(
        encoder_config, decoder_config
    )

    tokenizer = KoBertTokenizer()
    model = EncoderDecoderModel(config=config)
    ckpt = "model.pt"
    device = "cuda"

    model.load_state_dict(
        torch.load(
            f"saved/{ckpt}.{step}", map_location="cuda"
        ),
        strict=True,
    )

    model = model.half().eval().to(device)
    test_data = open("dataset/abstractive_test_v2.jsonl", "r").read().splitlines()
    submission = open(f"submission_{step}.csv", "w")

    test_set = []
    for data in test_data:
        data = json.loads(data)
        article_original = data["article_original"]
        article_original = " ".join(article_original)
        news_id = data["id"]
        test_set.append((news_id, article_original))

    for i, (news_id, text) in tqdm(enumerate(test_set)):
        tokens = tokenizer.encode_batch([text], max_length=512)
        generated = model.generate(
            input_ids=tokens["input_ids"].to(device),
            attention_mask=tokens["attention_mask"].to(device),
            use_cache=True,
            bos_token_id=tokenizer.token2idx["[CLS]"],
            eos_token_id=tokenizer.token2idx["[SEP]"],
            pad_token_id=tokenizer.token2idx["[PAD]"],
            num_beams=12,
            do_sample=False,
            temperature=1.0,
            no_repeat_ngram_size=4,
            bad_words_ids=[[tokenizer.token2idx["[UNK]"]]],
            length_penalty=1.5,
            max_length=512,
        )

        output = tokenizer.decode_batch(generated.tolist())[0]
        submission.write(f"{news_id},{output}" + "\n")
        print(news_id, output)


if __name__ == '__main__':
    inference()
