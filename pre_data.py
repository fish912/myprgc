from pathlib import Path
from ujson import loads, dump

__DATA_ROOT = Path() / 'Data' / 'duie'
TRAIN_DATA_PATH = __DATA_ROOT / 'train_triples.json'
TEST_DATA_PATH = __DATA_ROOT / 'duie_test2.json'
DEV_DATA_PATH = __DATA_ROOT / 'duie_dev.json'
SCHEMA_PATH = __DATA_ROOT / 'duie_schema.json'

SAVE_PATH = Path() / 'Data' / 'new_custom'
SAVE_PATH.mkdir(parents=True, exist_ok=True)


class ParseData:
    def __init__(self):
        self.prediction2id = {}
        self.id2predicate = {}
        self.predicate_length = 0
        self.label2idxSub = {"O": 0}
        self.label2IdxObj = {"O": 0}
        # 'O'，表示无实体或非命名实体。
        # B 表示实体头 H 表示 头实体。
        # I 表示实体中间 T 表示 尾实体。
        Label2IdxSub = {"B-H": 1, "I-H": 2, "O": 0}
        Label2IdxObj = {"B-T": 1, "I-T": 2, "O": 0}

    def prepare_data(self):
        # self.load_schema()
        self.load_train_data()

    def load_schema(self):
        num_p = 0
        obj = set()
        sub = set()
        with SCHEMA_PATH.open('r', encoding='utf-8', errors='replace') as f:
            while True:
                line = f.readline()
                if not line:
                    break
                data = loads(line)
                obj.add(data['object_type']['@value'])
                sub.add(data['subject_type'])
                self.prediction2id[data['predicate']] = num_p
                self.id2predicate[num_p] = data['predicate']
                num_p += 1
        for o in sorted(obj):
            self.label2IdxObj[f"B-T-{o}"] = len(self.label2IdxObj)
            self.label2IdxObj[f"I-T-{o}"] = len(self.label2IdxObj)
        for s in sorted(sub):
            self.label2idxSub[f"B-H-{s}"] = len(self.label2idxSub)
            self.label2idxSub[f"I-H-{s}"] = len(self.label2idxSub)
        with (SAVE_PATH / 'rel2id.json').open('w', encoding='utf-8', errors='replace') as f:
            dump(self.prediction2id, f)
        with (SAVE_PATH / 'id2rel.json').open('w', encoding='utf-8', errors='replace') as f:
            dump(self.id2predicate, f)
        with (SAVE_PATH / 'label2IdxObj.json').open('w', encoding='utf-8', errors='replace') as f:
            dump(self.label2IdxObj, f)
        with (SAVE_PATH / 'label2idxSub.json').open('w', encoding='utf-8', errors='replace') as f:
            dump(self.label2idxSub, f)
        self.predicate_length = num_p

    def load_train_data(self):
        all_data = []
        with TRAIN_DATA_PATH.open('r', encoding='utf-8', errors='replace') as f:
            while True:
                line = f.readline()
                if not line:
                    break
                data = loads(line)
                all_data.append(data)
                if len(all_data) > 50:
                    break
        new_data = []
        for d in all_data:
            text = d['text']
            spo_list = d['spo_list']
            new_data.append({
                'text': text,
                'triple_list': [[
                    one['subject'], one['predicate'], one['object']['@value']
                ] for one in spo_list]
            })
        with open('myprgc/Data/custom/train_triples.json', 'w', encoding='utf-8', errors='replace') as f:
            dump(new_data, f)
        return all_data


if __name__ == '__main__':
    p = ParseData()
    p.prepare_data()
