# /usr/bin/env python
# coding=utf-8
"""utils"""
import logging
import os
import shutil
import ujson
from pathlib import Path

import torch
from transformers import BertTokenizer, BertTokenizerFast, AutoTokenizer

# 'O'，表示无实体或非命名实体。
# B 表示实体头 H 表示 头实体。
# I 表示实体中间 T 表示 尾实体。
Label2IdxSub = {"B-H": 1, "I-H": 2, "O": 0}
Label2IdxObj = {"B-T": 1, "I-T": 2, "O": 0}


class Params:
    """参数定义
    """

    def __init__(self, ex_index='1', corpus_type='CUSTOM', **kwargs):
        self.root_path = Path(os.path.abspath(os.path.dirname(__file__)))
        self.data_dir = self.root_path / 'Data' / corpus_type
        self.ex_dir = self.root_path / 'experiments' / f'ex{ex_index}'
        self.ex_dir.mkdir(exist_ok=True, parents=True)
        self.model_dir = self.root_path / 'model' / f'ex{ex_index}'
        self.model_dir.mkdir(exist_ok=True, parents=True)
        self.bert_model_dir = self.root_path / 'pretrain_models'
        self.tokenizer_dir = self.root_path / 'tokenizer_pre'

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.n_gpu = torch.cuda.device_count()
        self.max_seq_length = kwargs.get('max_seq_length', 256)
        self.data_cache = kwargs.get('data_cache', False)  # 缓存影响加载的数据是否新旧，有缓存用旧的
        self.train_batch_size = kwargs.get('train_batch_size', 6)
        self.val_batch_size = kwargs.get('val_batch_size', 8)
        self.test_batch_size = kwargs.get('test_batch_size', 8)
        # self.train_batch_size = 6 if 'WebNLG' in corpus_type else 64
        # self.val_batch_size = 24
        # self.test_batch_size = 64
        # PRST parameters
        # load label2id
        with open(self.data_dir/'rel2id.json', 'r', encoding='utf-8') as f:
            json_data = ujson.load(f)
            self.rel2idx = json_data
        self.rel_num = len(self.rel2idx)
        with open(self.data_dir / 'label2idxObj.json', 'r', encoding='utf-8') as f:
            self.label2idObj = ujson.load(f)
        with open(self.data_dir / 'label2idxSub.json', 'r', encoding='utf-8') as f:
            self.label2idSub = ujson.load(f)
        self.seq_tag_size = len(self.label2idSub)
        # self.seq_tag_size_obj = len(self.label2idObj)
        self.seq_tag_size_obj = self.seq_tag_size
        # early stop strategy
        self.min_epoch_num = 20
        self.patience = 0.00001
        self.patience_num = 17

        # learning rate
        self.fin_tuning_lr = 1e-6  # 学习率
        self.downs_en_lr = 1e-5  # 下采样学习率
        self.clip_grad = 2.  # 梯度裁剪阈值,导数超过阈值就设置成阈值
        self.drop_prob = 0.3  # dropout 层的丢弃概率
        self.weight_decay_rate = 0.01  # 权重衰减的系数或速率
        self.warmup_prop = 0.1  # 学习率预热的比例
        self.gradient_accumulation_steps = 4  # 梯度累积的步骤数

        # self.tokenizer = BertTokenizerFast.from_pretrained(self.bert_model_dir, do_lower_case=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_dir, do_lower_case=True)
        self.best_val_f1 = 0.0

    def load(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = ujson.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        """保存配置到json文件
        """
        params = {}
        with open(json_path, 'w') as f:
            for k, v in self.__dict__.items():
                if isinstance(v, (str, int, float, bool, list, dict)):
                    params[k] = v
            ujson.dump(params, f, indent=4)


class RunningAverage:
    """A simple class that maintains the running average of a quantity

    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """

    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.steps += 1

    def __call__(self):
        return self.total / float(self.steps)


def set_logger(save=False, log_path=None):
    """Set the logger to log info in terminal and file `log_path`.

    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.

    Example:
    ```
    logging.info("Starting training...")
    ```

    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if save and not os.path.exists(os.path.dirname(log_path)):
        os.makedirs(os.path.dirname(log_path))

    if not logger.handlers:
        if save:
            # Logging to a file
            file_handler = logging.FileHandler(log_path)
            file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
            logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


def save_checkpoint(state, is_best, checkpoint):
    """Saves model and training parameters at checkpoint + 'last.pth.tar'. If is_best==True, also saves
    checkpoint + 'best.pth.tar'

    Args:
        state: (dict) contains the entire model, may contain other keys such as epoch, optimizer
        is_best: (bool) True if it is the best model seen till now
        checkpoint: (string) folder where parameters are to be saved
    """
    filepath = os.path.join(checkpoint, 'last.pth.tar')
    if not os.path.exists(checkpoint):
        print("Checkpoint Directory does not exist! Making directory {}".format(checkpoint))
        os.makedirs(checkpoint)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'best.pth.tar'))


def load_checkpoint(checkpoint, optimizer=True):
    """Loads entire model from file_path. If optimizer is True, loads
    optimizer assuming it is present in checkpoint.

    Args:
        checkpoint: (string) filename which needs to be loaded
        optimizer: (bool) resume optimizer from checkpoint
    """
    if not os.path.exists(checkpoint):
        raise ValueError("File doesn't exist {}".format(checkpoint))
    checkpoint = torch.load(checkpoint, map_location=torch.device('cpu'))

    if optimizer:
        return checkpoint['model'], checkpoint['model_config'], checkpoint['optim'], checkpoint.get('f1', 0)
    return checkpoint['model'], checkpoint['model_config']
