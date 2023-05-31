import os

if __name__ == '__main__':
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
            new_line = '%s,%s'%(pth, cls//2)
            if cls % 2 == 1:
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
            new_line = '%s,%s'%(pth, cls//2)
            if cls % 2 == 0:
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
            new_line = '%s,%s'%(pth, cls//2)
            if cls % 2 == 0:
                output_f.write(new_line)
                output_f.write('\n')

 
        



