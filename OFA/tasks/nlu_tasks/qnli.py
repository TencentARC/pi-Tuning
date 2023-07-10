# flake8: noqa

# Copyright 2022 The OFA-Sys Team. 
# All rights reserved.
# This source code is licensed under the Apache 2.0 license 
# found in the LICENSE file in the root directory.

import json
import logging
import math
from dataclasses import dataclass, field
from typing import Optional

import torch
from fairseq import metrics
from fairseq.dataclass import ChoiceEnum
from fairseq.tasks import register_task

from data import data_utils
from tasks.ofa_task import OFATask, OFAConfig
from data.nlu_data.qnli_dataset import QNLIDataset
from data.file_dataset import FileDataset
from utils.trie import Trie

logger = logging.getLogger(__name__)


@dataclass
class QNLIConfig(OFAConfig):
    ans2label_dict: Optional[str] = field(
        default='{"no": 0, "yes": 1}',
        metadata={"help": 'answer to label dict'},
    )
    prompt_type: ChoiceEnum(["none", "src", "prev_output"]) = field(
        default="none",
        metadata={"help": "decoder prompt"},
    )
    valid_batch_size: int = field(
        default=20,
        metadata={"help": "valid batch size per step"},
    )


@register_task("qnli", dataclass=QNLIConfig)
class QNLITask(OFATask):
    def __init__(self, cfg: QNLIConfig, src_dict, tgt_dict):
        super().__init__(cfg, src_dict, tgt_dict)
        self.ans2label_dict = json.loads(self.cfg.ans2label_dict)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        paths = self.cfg.data.split(',')
        assert len(paths) > 0

        if split == 'train':
            file_path = paths[(epoch - 1) % (len(paths) - 1)]
        else:
            file_path = paths[-1]
        dataset = FileDataset(file_path, self.cfg.selected_cols)

        self.datasets[split] = QNLIDataset(
            split,
            dataset,
            self.bpe,
            self.src_dict,
            self.tgt_dict,
            max_src_length=self.cfg.max_src_length,
            max_tgt_length=self.cfg.max_tgt_length,
            constraint_trie=self.constraint_trie,
            prompt_type=self.cfg.prompt_type
        )

    def build_model(self, cfg):
        model = super().build_model(cfg)
        self.constraint_trie = Trie(self.tgt_dict.eos())
        for i, answer in enumerate(self.ans2label_dict.keys()):
            answer_item = self.tgt_dict.encode_line(
                line=self.bpe.encode(' ' + answer),
                add_if_not_exist=False,
                append_eos=False
            ).long()
            self.constraint_trie.insert([self.tgt_dict.bos()] + answer_item.tolist() + [self.tgt_dict.eos()])
        tgt_list = []
        prev_output_list = []
        self.index2ans = {}
        self.ans2index = {}
        for i, answer in enumerate(self.ans2label_dict.keys()):
            answer_item = self.tgt_dict.encode_line(
                line=self.bpe.encode(' ' + answer),
                add_if_not_exist=False,
                append_eos=False
            ).long()
            self.constraint_trie.insert([self.tgt_dict.bos()] + answer_item.tolist() + [self.tgt_dict.eos()])
            tgt_list += [torch.cat([answer_item, torch.LongTensor([self.tgt_dict.eos()])])]
            prev_output_list += [torch.cat([torch.LongTensor([self.tgt_dict.bos()]), answer_item])]
            self.index2ans[i] = answer
            self.ans2index[answer] = i

        constraint_mask_list = []
        for prev_output_item in prev_output_list:
            constraint_mask = torch.zeros((len(prev_output_item), len(self.tgt_dict))).bool()
            for i in range(len(prev_output_item)):
                constraint_prefix_token = prev_output_item[:i + 1].tolist()
                constraint_nodes = self.constraint_trie.get_next_layer(constraint_prefix_token)
                constraint_mask[i][constraint_nodes] = True
            constraint_mask_list.append(constraint_mask)

        eos = self.src_dict.eos()
        pad = self.src_dict.pad()
        self.valid_tgt_list = []
        self.valid_prev_output_list = []
        self.valid_constraint_masks_list = []
        for i in range(0, len(tgt_list), self.cfg.valid_batch_size):
            tgt_item = tgt_list[i:i + self.cfg.valid_batch_size]
            prev_output_item = prev_output_list[i:i + self.cfg.valid_batch_size]
            constrain_mask = constraint_mask_list[i:i + self.cfg.valid_batch_size]
            self.valid_tgt_list.append(
                data_utils.collate_tokens(tgt_item, pad_idx=pad, eos_idx=eos, left_pad=False)
            )
            self.valid_prev_output_list.append(
                data_utils.collate_tokens(prev_output_item, pad_idx=pad, eos_idx=eos, left_pad=False)
            )
            self.valid_constraint_masks_list.append(
                data_utils.collate_tokens(constrain_mask, pad_idx=pad, left_pad=False)
            )
        return model

    def build_generator(
        self, models, args, seq_gen_cls=None, extra_gen_cls_kwargs=None, prefix_allowed_tokens_fn=None,
    ):
        seq_generator = super().build_generator(models, args, seq_gen_cls, extra_gen_cls_kwargs, prefix_allowed_tokens_fn)
        seq_generator.constraint_trie = self.constraint_trie

        return seq_generator

    def valid_step(self, sample, model, criterion):
        loss, sample_size, logging_output = super().valid_step(sample, model, criterion)
        model.eval()
        with torch.no_grad():
            net_output = model(**sample["net_input"])
            net_output[0].masked_fill_(~sample["constraint_masks"], -math.inf)
            last_token_ids = sample["net_input"]["prev_output_tokens"].ne(self.src_dict.pad()).sum(1, keepdim=True) - 1
            logits = net_output[0].gather(1, last_token_ids.unsqueeze(2).expand(-1, -1, net_output[0].size(2)))
            logits = logits.squeeze(1)
            predicts = logits.argmax(1).tolist()
            hyps = [self.bpe.decode(self.src_dict[predict]).strip() for predict in predicts]
            scores = [ref_dict.get(hyp, 0) for ref_dict, hyp in zip(sample['ref_dict'], hyps)]
        logging_output["_score_sum"] = sum(scores)
        logging_output["_score_cnt"] = len(scores)

        return loss, sample_size, logging_output

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)

        def sum_logs(key):
            import torch
            result = sum(log.get(key, 0) for log in logging_outputs)
            if torch.is_tensor(result):
                result = result.cpu()
            return result

        def compute_acc(meters):
            score = meters["_score_sum"].sum / meters["_score_cnt"].sum
            score = score if isinstance(score, float) else score.item()
            return round(score, 4)

        if sum_logs("_score_cnt") > 0:
            metrics.log_scalar("_score_sum", sum_logs("_score_sum"))
            metrics.log_scalar("_score_cnt", sum_logs("_score_cnt"))
            metrics.log_derived("acc", compute_acc)
