import os
import numpy as np
import json

if __name__ == '__main__':
    lines = open('./raw_ann.txt', 'r').readlines()
    root = '/share_io02_ssd/jia/kinetics-600-compress/videos'
    k400_cls_set = set(json.load(open('../k400-cls2index.json', 'r')).keys())
    
    k600_openset_cls_set = {}

    # achieve cls_index mapping file
    cls_dir = {}
    for line in lines[1:601]:
        cls_dir[line.strip()] = None
    
    cls_sorted_list = sorted(cls_dir)
    
    cls2index_mapping = {}
    index2cls_mapping = {}

    for idx, cls in enumerate(cls_sorted_list):
        cls2index_mapping[cls] = idx
        index2cls_mapping[idx] = cls

    json.dump(cls2index_mapping, open('../k600-cls2index.json', 'w'))
    json.dump(index2cls_mapping, open('../k600-index2cls.json', 'w'))
        
    f = open('./test.csv', 'w')
    for idx, filename in enumerate(cls_sorted_list):
        cls_id = idx
        cls_name = filename.strip().replace(' ', '_')
        videoname_list = os.listdir(
                os.path.join(root, 'val', cls_name)
            )
        for vn in videoname_list:
            f.write('%s,%d'%(os.path.join('val', cls_name, vn),cls_id))
            f.write('\n')
    
    f.close()
    
    ############
    openset_cls_list = []
    f = open('./test_openset.csv', 'w')
    cls_index = 0
    for filename in cls_sorted_list:
        cls_id = cls_index
        cls_name = filename.strip().replace(' ', '_')
        
        if filename in k400_cls_set:
            continue
        openset_cls_list.append(filename)

        videoname_list = os.listdir(
                 os.path.join(root, 'val', cls_name)
                )
        for vn in videoname_list:
            f.write('%s,%d'%(os.path.join('val', cls_name, vn),cls_id))
            f.write('\n')
        cls_index += 1
    
    f.close()
    
    # openset json save 
    open_cls2index_mapping = {}
    open_index2cls_mapping = {}
    for idx, cls in enumerate(openset_cls_list):
        open_cls2index_mapping[cls] = idx
        open_index2cls_mapping[idx] = cls

    json.dump(open_cls2index_mapping, open('../k600-openset-cls2index.json', 'w'))
    json.dump(open_index2cls_mapping, open('../k600-openset-index2cls.json', 'w'))
    
    
    


