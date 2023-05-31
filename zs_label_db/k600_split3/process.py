import os
import numpy as np
import json

if __name__ == '__main__':
    lines = open('./resource/raw_ann.txt', 'r').readlines()
    root = '/share_io02_ssd/jia/kinetics-600-compress/videos'
    # k400_cls_set = set(json.load(open('../../label_db/k400-cls2index.json', 'r')).keys())
    split_partition = 2
    all_cls_list = [i['word'] for i in json.load(open('./resource/classes620_label_defn.json', 'r'))]
    openidx = json.load(open('./resource/tst_class_idxs.json', 'r'))[split_partition]
    openset_cls_list = [all_cls_list[i] for i in openidx]
    
    # all k600 cls
    k600_cls_dir = {}
    for line in lines[1:601]:
        k600_cls_dir[line.strip()] = None
    k600_cls_sorted_list = sorted(k600_cls_dir)
    
    # testing
    for cls in openset_cls_list:
        if cls not in k600_cls_sorted_list:
            print("problem exists")
            exit()
    
    openset_cls_list = sorted(openset_cls_list)
    ############
    f = open('./test.csv', 'w')
    
    for cls_id, filename in enumerate(openset_cls_list):
        cls_name = filename.strip().replace(' ', '_') 

        videoname_list = os.listdir(
                 os.path.join(root, 'val', cls_name)
                )
        if not videoname_list:
            print("empty dir, wrong")
            exit()

        for vn in videoname_list:
            f.write('%s,%d'%(os.path.join('val', cls_name, vn),cls_id))
            f.write('\n')
        
    # openset json save 
    open_cls2index_mapping = {}
    open_index2cls_mapping = {}
    for idx, cls in enumerate(openset_cls_list):
        open_cls2index_mapping[cls] = idx
        open_index2cls_mapping[idx] = cls

    json.dump(open_cls2index_mapping, open('./k600-openset-cls2index.json', 'w'))
    json.dump(open_index2cls_mapping, open('./k600-openset-index2cls.json', 'w'))
    
    
    


