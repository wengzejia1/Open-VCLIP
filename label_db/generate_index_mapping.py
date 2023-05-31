import numpy as np
import os
import json

def parse_idx2label(root_path, target_path):
    lines = open(root_path, 'r').readlines()
    index_mapping = {}

    for line in lines:
        text, index = line.split(',')
        text = text.split('/')[1]        
        index_mapping[int(index)] = text

    json.dump(index_mapping, open(target_path, 'w'))


def parse_label2idx(root_path, target_path):
    lines = open(root_path, 'r').readlines()
    index_mapping = {}

    for line in lines:
        text, index = line.split(',')
        text = text.split('/')[1]        
        # index_mapping[int(index)] = text
        index_mapping[text] = int(index)

    json.dump(index_mapping, open(target_path, 'w'))


parse_idx2label('./weng_full_splits/train.csv', './k400-index2cls.json')
parse_idx2label('./weng_partial_splits/train.csv', './k200-closeset-index2cls.json')
parse_idx2label('./weng_partial_splits/test_openset.csv', './k200-openset-index2cls.json')

parse_label2idx('./weng_full_splits/train.csv', './k400-cls2index.json')
parse_label2idx('./weng_partial_splits/train.csv', './k200-closeset-cls2index.json')
parse_label2idx('./weng_partial_splits/test_openset.csv', './k200-openset-cls2index.json')


