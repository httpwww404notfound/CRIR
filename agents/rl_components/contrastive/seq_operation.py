import torch
import functools
import numpy as np
from .data_argument import Crop

class seq_operate(object):

    def __init__(self, max_len):
        super().__init__()
        self.base_transform = Crop()
        self.max_len = max_len

    def de_padding(self, input_seqs):
        output_seqs = []
        for seq in input_seqs:
            de_padded = self._de_padding(seq)
            output_seqs.append(de_padded)
        return output_seqs

    def _de_padding(self, input_seq):
        start_idx = 0
        for idx, i in enumerate(input_seq):
            if i != 0:
                start_idx = idx
                break
        return input_seq[start_idx:]

    def padding(self, input_seqs, max_len = None, pad_value=0):
        output_seqs = []
        for i, seq in enumerate(input_seqs):
            padding_seq = self._padding(seq, max_len=max_len, pad_value=pad_value)
            output_seqs.append(padding_seq)
        return output_seqs

    def reverse_padding(self, input_seqs, max_len = None, pad_value=0):
        output_seqs = []
        for i, seq in enumerate(input_seqs):
            padding_seq = self._reverse_padding(seq, max_len=max_len, pad_value=pad_value)
            output_seqs.append(padding_seq)
        return output_seqs

    def _padding(self, input_ids, max_len = None, pad_value=0):
        if max_len is None:
            max_len = self.max_len

        pad_len = max_len - len(input_ids)
        output_ids = [pad_value] * pad_len + input_ids
        output_ids = output_ids[-max_len:]

        assert len(output_ids) == max_len

        return output_ids

    def _reverse_padding(self, input_ids, max_len = None, pad_value=0):
        if max_len is None:
            max_len = self.max_len

        pad_len = max_len - len(input_ids)
        output_ids = input_ids + [pad_value] * pad_len
        output_ids = output_ids[:max_len]

        assert len(output_ids) == max_len

        return output_ids

    def one_pair_data_augmentation(self, input_ids, pad_value=0):
        """
        provides two positive samples given one sequence
        """
        augmented_seqs = []
        for i in range(2):
            augmented_input_ids = self.base_transform(input_ids)
            pad_len = self.max_len - len(augmented_input_ids)
            augmented_input_ids = [pad_value] * pad_len + augmented_input_ids

            augmented_input_ids = augmented_input_ids[-self.max_len:]

            assert len(augmented_input_ids) == self.max_len

            augmented_seqs.append(augmented_input_ids)
        return augmented_seqs

    def items_contrast_ids(self, input_ids, rank):
        """
        generate augmentation pairs from sequences split by attention weight
        input_ids: list of unbalanced list of item ids
        rank: list batch_size * max_len
        top_k: bool
        """
        sorted_ids = []
        for i, seq in enumerate(input_ids):
            index_pair = list(enumerate(seq))
            ranked_pair = sorted(index_pair, key=lambda x:rank[i][x[0]], reverse=True)
            ranked = [i[1] for i in ranked_pair]
            sorted_ids.append(ranked)

        sampled_rank_ids = []
        k_list = []
        for index, seq in enumerate(sorted_ids):
            k = 1 + np.random.choice(len(seq)//2)
            k_list.append(k)
            items = [seq[0]] + [seq[k]] + seq[len(seq)//2:]
            sampled_rank_ids.append(items)


        return sampled_rank_ids, k_list
        