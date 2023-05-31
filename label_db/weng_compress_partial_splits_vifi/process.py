import os

if __name__ == '__main__':
    f = open('../weng_compress_full_splits/train.csv', 'r').readlines()
    cls_freq = {}
    for line in f:
        cls = int(line.split(",")[1])
        if cls not in cls_freq:
            cls_freq[cls] = 0
        cls_freq[cls] += 1
    
    cls_freq_list = []
    for cls_id in range(len(cls_freq)):
        cls_freq_list.append((cls_id,cls_freq[cls_id]))
    
    cls_freq_list = sorted(cls_freq_list, key = lambda x:x[1], reverse=True)
    closeset = [i[0] for i in cls_freq_list[:200]]
    openset = [i[0] for i in cls_freq_list[200:]]
    
    close_mapping = {}
    open_mapping = {}

    for i, cls in enumerate(closeset):
        close_mapping[cls] = i
    for i, cls in enumerate(openset):
        open_mapping[cls] = i

    # process test csv
    with open('test_openset.csv', 'w') as output_f:
        filename = '../weng_compress_full_splits/test.csv'
        f = open(filename, 'r')
        lines = f.readlines()
        f_num = len(lines)
        
        for idx, line in enumerate(lines):
            line = line.strip()
            pth, cls = line.split(",")
            cls = int(cls)
            
            if cls in openset:
                new_line = '%s,%s'%(pth, open_mapping[cls])
                output_f.write(new_line)
                output_f.write('\n')  
    
    with open('test.csv', 'w') as output_f:
        filename = '../weng_compress_full_splits/test.csv'
        f = open(filename, 'r')
        lines = f.readlines()
        f_num = len(lines)
        for idx, line in enumerate(lines):
            line = line.strip()
            pth, cls = line.split(",")
            cls = int(cls)

            if cls in closeset:
                new_line = '%s,%s'%(pth, close_mapping[cls])
                output_f.write(new_line)
                output_f.write('\n')
    
    # process train csv
    with open('train.csv', 'w') as output_f:
        filename = '../weng_compress_full_splits/train.csv'
        f = open(filename, 'r')
        lines = f.readlines()
        f_num = len(lines)
        for idx, line in enumerate(lines):
            line = line.strip()
            pth, cls = line.split(",")
            cls = int(cls)
            
            if cls in closeset:
                new_line = '%s,%s'%(pth, close_mapping[cls])
                output_f.write(new_line)
                output_f.write('\n')

 
        



