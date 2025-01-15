# !/usr/bin/env python3
# -*- coding: utf-8 -*-
import random
import re
import numpy as np
from collections import defaultdict

import tqdm
from ujson import loads


class InputExample(object):
    """a single set of samples of data
    """

    def __init__(self, text, pair_list, re_list, relId_pair):
        self.text = text
        self.pair_list = pair_list
        self.re_list = re_list
        self.relId_pair = relId_pair


class InputFeatures(object):
    """
    Desc:
        a single set of features of data
    """

    def __init__(self,
                 input_tokens,
                 input_ids,
                 attention_mask,
                 seq_tag=None,
                 corres_tag=None,
                 relation=None,
                 triples=None,
                 rel_tag=None
                 ):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.seq_tag = seq_tag
        self.corres_tag = corres_tag
        self.relation = relation
        self.triples = triples
        self.rel_tag = rel_tag


def clear_text(text):
    return re.compile(r'[@#$%^&*()<>?:…"\'‘’“”、【】·~!¥（）`—_《》+\\|{}]+').sub('', text)


def add_vocab(tokenizer, text, new_tokens):
    t = tokenizer(text, return_offsets_mapping=True)
    new_token = [text[t.offset_mapping[i][0]] for i, v in enumerate(t.tokens()) if v == '[UNK]']
    if len(new_token) != 0:
        new_tokens.extend(new_token)


def read_examples(data_dir, tokenizer, data_sign, rel2idx):
    """load data to InputExamples
    """
    examples = []
    new_tokens = []

    # read src data
    with open(data_dir / f'{data_sign}_triples.json', "r", encoding='utf-8') as f:
        while True:
            line = f.readline()
            if not line:
                break
            data = loads(line)
            text = clear_text(data['text'])
            add_vocab(tokenizer, text, new_tokens)
            relId_pair = defaultdict(list)
            pair_list = []
            re_list = []

            for triple in data['spo_list']:
                t0, t1, t2 = triple['subject'], triple['predicate'], triple['object']['@value']
                t0_attr, t2_attr = triple['subject_type'], triple['object_type']['@value']
                t0 = clear_text(t0)
                t1 = clear_text(t1)
                t2 = clear_text(t2)
                pair_list.append([t0, t2, t0_attr, t2_attr])
                re_list.append(rel2idx[t1])
                relId_pair[rel2idx[t1]].append((t0, t2, t0_attr, t2_attr))
            example = InputExample(text=text, pair_list=pair_list, re_list=re_list, relId_pair=relId_pair)
            examples.append(example)
            if len(examples) >= 30000 and data_sign == "train":  # ex2
                break
            elif len(examples) >= 8000 and data_sign == "val":  # ex2
                break
    if len(new_tokens) != 0:
        tokenizer.add_tokens(new_tokens)
    print("InputExamples:", len(examples))
    return examples


def find_head_idx(source, target):
    target_len = len(target)
    for i in range(len(source)):
        if source[i: i + target_len] == target:
            return i
    return -1


def _get_so_head(en_pair, tokenizer, text_tokens):
    sub = tokenizer.tokenize(en_pair[0])
    obj = tokenizer.tokenize(en_pair[1])
    sub_head = find_head_idx(source=text_tokens, target=sub)
    if sub == obj:
        obj_head = find_head_idx(source=text_tokens[sub_head + len(sub):], target=obj)
        if obj_head != -1:
            obj_head += sub_head + len(sub)
        else:
            obj_head = sub_head
    else:
        obj_head = find_head_idx(source=text_tokens, target=obj)
    return sub_head, obj_head, sub, obj


def convert(example, max_text_len, tokenizer, rel2idx, data_sign, ex_params, label2id_obj, label2id_sub):
    """convert function
    """
    text_tokens = tokenizer.tokenize(example.text)
    # cut off
    if len(text_tokens) > max_text_len:
        text_tokens = text_tokens[:max_text_len]

    # token to id
    input_ids = tokenizer.convert_tokens_to_ids(text_tokens)
    attention_mask = [1] * len(input_ids)
    # zero-padding up to the sequence length
    if len(input_ids) < max_text_len:
        pad_len = max_text_len - len(input_ids)
        # token_pad_id=0
        input_ids += [0] * pad_len
        attention_mask += [0] * pad_len

    # train data
    if data_sign == 'train':
        # construct tags of correspondence and relation
        corres_tag = np.zeros((max_text_len, max_text_len))
        rel_tag = len(rel2idx) * [0]
        for en_pair, rel in zip(example.pair_list, example.re_list):
            # get sub and obj head
            sub_head, obj_head, _, _ = _get_so_head(en_pair[:2], tokenizer, text_tokens)
            # construct relation tag
            rel_tag[rel] = 1
            if sub_head != -1 and obj_head != -1:
                corres_tag[sub_head][obj_head] = 1

        sub_feats = []
        # positive samples
        for rel, pair_list in example.relId_pair.items():
            # init
            tags_sub = max_text_len * [label2id_obj['O']]
            tags_obj = max_text_len * [label2id_sub['O']]
            for pair in pair_list:
                # get sub and obj head
                sub, obj, sub_attr, obj_attr = pair
                sub_head, obj_head, sub, obj = _get_so_head((sub, obj), tokenizer, text_tokens)
                # UNK 影响准确性
                if '[UNK]' in sub or '[UNK]' in obj:
                    print(sub, obj, 'WARN: *!@#$-----!@#$-----!@#$-----!@#$-----!@#$-----!@#$-----!@#$-----!@#$')
                if sub_head != -1 and obj_head != -1:
                    if sub_head + len(sub) <= max_text_len:
                        tags_sub[sub_head] = label2id_sub[f'B-H-{sub_attr}']
                        tags_sub[sub_head + 1:sub_head + len(sub)] = (len(sub) - 1) * [label2id_sub[f'I-H-{sub_attr}']]
                        tags_sub[sub_head + 1:sub_head + len(sub)] = (len(sub) - 1) * [label2id_sub[f'I-H-{sub_attr}']]
                    if obj_head + len(obj) <= max_text_len:
                        tags_obj[obj_head] = label2id_obj[f'B-T-{obj_attr}']
                        tags_obj[obj_head + 1:obj_head + len(obj)] = (len(obj) - 1) * [label2id_obj[f'I-T-{obj_attr}']]
                        tags_obj[obj_head + 1:obj_head + len(obj)] = (len(obj) - 1) * [label2id_obj[f'I-T-{obj_attr}']]
            seq_tag = [tags_sub, tags_obj]

            # sanity check
            assert len(input_ids) == len(tags_sub) == len(tags_obj) == len(
                attention_mask) == max_text_len, f'length is not equal!!'
            sub_feats.append(InputFeatures(
                input_tokens=text_tokens,
                input_ids=input_ids,
                attention_mask=attention_mask,
                corres_tag=corres_tag,  # max_text_len*max_text_len
                seq_tag=seq_tag,  # 1*2*max_text_len
                relation=rel,  # relId
                rel_tag=rel_tag  # 1*rel_len
            ))
        # relation judgement ablation
        if not ex_params['ensure_rel']:
            # negative samples
            neg_rels = set(rel2idx.values()).difference(set(example.re_list))
            neg_rels = random.sample(neg_rels, k=ex_params['num_negs'])
            for neg_rel in neg_rels:
                # init
                seq_tag = max_text_len * [label2id_sub['O']]
                # sanity check
                assert len(input_ids) == len(seq_tag) == len(attention_mask) == max_text_len, f'length is not equal!!'
                seq_tag = [seq_tag, seq_tag]
                sub_feats.append(InputFeatures(
                    input_tokens=text_tokens,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    corres_tag=corres_tag,
                    seq_tag=seq_tag,
                    relation=neg_rel,
                    rel_tag=rel_tag
                ))
    # val and test data
    else:
        triples = []
        for rel, entity in zip(example.re_list, example.pair_list):
            # get sub and obj head
            sub_head, obj_head, sub, obj = _get_so_head(entity[:2], tokenizer, text_tokens)
            if sub_head != -1 and obj_head != -1:
                h_chunk = ('H', sub_head, sub_head + len(sub), entity[-2])
                t_chunk = ('T', obj_head, obj_head + len(obj), entity[-1])
                triples.append((h_chunk, t_chunk, rel))
        sub_feats = [
            InputFeatures(
                input_tokens=text_tokens,
                input_ids=input_ids,
                attention_mask=attention_mask,
                triples=triples
            )
        ]

    # get sub-feats
    return sub_feats


def convert_examples_to_features(params, examples, tokenizer, rel2idx, data_sign, ex_params):
    """convert examples to features.
    :param examples (List[InputExamples])
    """
    max_text_len = params.max_seq_length
    # multi-process
    # with Pool(10) as p:
    #     convert_func = functools.partial(convert, max_text_len=max_text_len, tokenizer=tokenizer, rel2idx=rel2idx,
    #                                      data_sign=data_sign, ex_params=ex_params)
    #     features = p.map(func=convert_func, iterable=examples)
    features = []
    for example in tqdm.tqdm(examples):
        f = convert(example, max_text_len=max_text_len, tokenizer=tokenizer, rel2idx=rel2idx, data_sign=data_sign,
                    ex_params=ex_params, label2id_obj=params.label2idObj, label2id_sub=params.label2idSub)
        features.extend(f)

    return features
