from typing import List
from transformers import (
    EncoderDecoderModel,
    BertConfig,
    EncoderDecoderConfig,
    BertModel, BertTokenizer,
)
from transformers.modeling_bart import shift_tokens_right
from kobert_transformers import get_tokenizer
from lightning_base import LightningBase
import torch


class Bert2Bert(LightningBase):
    def __init__(
            self,
            model_save_path: str,
            batch_size: int,
            num_gpus: int,
            max_len: int = 512,
            lr: float = 3e-5,
            weight_decay: float = 1e-4,
            save_step_interval: int = 1000,
            accelerator: str = "ddp",
            precision: int = 16,
            use_amp: bool = True,
    ) -> None:
        super(Bert2Bert, self).__init__(
            model_save_path=model_save_path,
            max_len=max_len,
            batch_size=batch_size,
            num_gpus=num_gpus,
            lr=lr,
            weight_decay=weight_decay,
            save_step_interval=save_step_interval,
            accelerator=accelerator,
            precision=precision,
            use_amp=use_amp,
        )
        encoder_config = BertConfig.from_pretrained("monologg/kobert")
        decoder_config = BertConfig.from_pretrained("monologg/kobert")
        config = EncoderDecoderConfig.from_encoder_decoder_configs(
            encoder_config, decoder_config
        )

        self.model = EncoderDecoderModel(config)
        self.tokenizer = KoBertTokenizer()

        state_dict = BertModel.from_pretrained("monologg/kobert").state_dict()
        self.model.encoder.load_state_dict(state_dict)
        self.model.decoder.bert.load_state_dict(state_dict, strict=False)
        # cross attention이랑 lm head는 처음부터 학습

    def training_step(self, batch, batch_idx):
        src, tgt = batch[0], batch[1]
        src_input = self.tokenizer.encode_batch(src, max_length=self.max_len)
        tgt_input = self.tokenizer.encode_batch(tgt, max_length=self.max_len)

        input_ids = src_input["input_ids"].to(self.device)
        attention_mask = src_input["attention_mask"].to(self.device)
        labels = tgt_input["input_ids"].to(self.device)
        decoder_input_ids = shift_tokens_right(
            labels, self.tokenizer.token2idx["[PAD]"]
        )

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
        )

        lm_logits = outputs[0]
        loss_fn = torch.nn.CrossEntropyLoss(
            ignore_index=self.tokenizer.token2idx["[PAD]"]
        )

        lm_loss = loss_fn(lm_logits.view(-1, lm_logits.shape[-1]), labels.view(-1))
        self.save_model()
        return {"loss": lm_loss}


class KoBertTokenizer(object):
    def __init__(self):
        self.tokenizer = get_tokenizer()
        self.token2idx = self.tokenizer.token2idx
        self.idx2token = {v: k for k, v in self.token2idx.items()}

    def encode_batch(self, x: List[str], max_length):
        max_len = 0
        result_tokenization = []

        for i in x:
            tokens = self.tokenizer.encode(i, max_length=max_length, truncation=True)
            result_tokenization.append(tokens)

            if len(tokens) > max_len:
                max_len = len(tokens)

        padded_tokens = []
        for tokens in result_tokenization:
            padding = (torch.ones(max_len) * self.token2idx["[PAD]"]).long()
            padding[: len(tokens)] = torch.tensor(tokens).long()
            padded_tokens.append(padding.unsqueeze(0))

        padded_tokens = torch.cat(padded_tokens, dim=0).long()
        mask_tensor = torch.ones(padded_tokens.size()).long()

        attention_mask = torch.where(
            padded_tokens == self.token2idx["[PAD]"], padded_tokens, mask_tensor * -1
        ).long()
        attention_mask = torch.where(
            attention_mask == -1, attention_mask, mask_tensor * 0
        ).long()
        attention_mask = torch.where(
            attention_mask != -1, attention_mask, mask_tensor
        ).long()

        return {
            "input_ids": padded_tokens.long(),
            "attention_mask": attention_mask.long(),
        }

    def decode(self, tokens):
        # remove special tokens
        # unk, pad, cls, sep, mask
        tokens = [token for token in tokens
                  if token not in [0, 1, 2, 3, 4]]

        decoded = [self.idx2token[token] for token in tokens]
        if "▁" in decoded[0] and "▁" in decoded[1]:
            # fix decoding bugs
            tokens = tokens[1:]

        return self.tokenizer.decode(tokens)

    def decode_batch(self, list_of_tokens):
        return [self.decode(tokens) for tokens in list_of_tokens]
