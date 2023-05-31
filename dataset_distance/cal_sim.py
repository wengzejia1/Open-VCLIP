import torch
# import slowfast.model.clip as clip
import clip


def cal_dataset_dis(csv1, csv2, remove_overlap=False, split_half=False):
    
    if split_half:
        k4 = open(csv1, 'r').readlines()[1:] 
    else: 
        k4 = open(csv1, 'r').readlines()[1:]
        k6 = open(csv2, 'r').readlines()[1:]
    k4_d = set()
    k6_d = set()
    
    if split_half:
        for idx, line in enumerate(k4):
            if idx % 2 == 0:
                k4_d.add(line.strip().split(',')[1])
            else:
                k6_d.add(line.strip().split(',')[1])
    else:
        for line in k4:
            k4_d.add(line.strip().split(',')[1])

        for line in k6:
            k6_d.add(line.strip().split(',')[1])

    if remove_overlap:
        unseen_cls = list(k6_d - k4_d)
    else:
        unseen_cls = list(k6_d)
    
    closeset_cls = list(k4_d)

    device = 'cuda'

    model, preprocess = clip.load("ViT-B/16", device=device)


    with torch.no_grad():
        closeset_feat = model.encode_text(
                    clip.tokenize(
                        ['%s'%c for c in closeset_cls]
                        ).to(device)
                    )

        unseenset_feat = model.encode_text(
                    clip.tokenize(
                        ['%s'%c for c in unseen_cls]
                        ).to(device)
                    )

    closeset_feat /= closeset_feat.norm(dim=-1, keepdim=True)
    unseenset_feat /= unseenset_feat.norm(dim=-1, keepdim=True)

    # print(['a photo of %s'%c for c in closeset_cls])  
    similarity_score = unseenset_feat @ closeset_feat.T
    
    print("==================")
    if remove_overlap:
        print("REMOVE OVERLAP")
    
    if split_half:
        print("split half distance in dataset %s"%csv1)
    else:
        print("distance between %s and %s"%(csv1, csv2))
    print('mean of max: %.4f'%similarity_score.max(dim=-1)[0].mean().item())
    print(similarity_score.max(dim=-1)[0].max().item())
    print(similarity_score.max(dim=-1)[0].min().item())


cal_dataset_dis('./kinetics_400_labels.csv', './kinetics_600_labels.csv')
cal_dataset_dis('./kinetics_400_labels.csv', './kinetics_600_labels.csv', remove_overlap=True)

cal_dataset_dis('./kinetics_400_labels.csv', './hmdb_51_labels.csv')
cal_dataset_dis('./kinetics_400_labels.csv', './hmdb_51_labels.csv', remove_overlap=True)

cal_dataset_dis('./kinetics_400_labels.csv', './ucf_101_labels.csv')
cal_dataset_dis('./kinetics_400_labels.csv', './ucf_101_labels.csv', remove_overlap=True)

cal_dataset_dis('./kinetics_400_labels.csv', './somethingsomething.csv', remove_overlap=True)

cal_dataset_dis('./kinetics_400_labels.csv', None, remove_overlap=False, split_half=True)


