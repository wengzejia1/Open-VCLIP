import torch
import torch.nn as nn
from . import clip

from .build import MODEL_REGISTRY
import os
import numpy as np
import json

@MODEL_REGISTRY.register()
class BasicClipVideo(nn.Module):
    """
    Clip visual encoder for space feature extraction. Adding various temporal fusion type.
    """
    def __init__(self, cfg):
        """
        The `__init__` method of any subclass should also contain these
            arguments.
        Args:
            cfg (CfgNode): model building configs, details are in the
            comments of the config file.
        """
        super(BasicClipVideo, self).__init__()
        self.cfg = cfg
        self.num_pathways = 1
        self._construct_network(cfg)
        
        # text encoder
        self.model.eval() 
        
        if not cfg.TEST.OPENSET:
            # self.text_dict = self.text_prompt(os.path.join(cfg.DATA.PATH_TO_DATA_DIR, 'train.csv'))
            self.text_dict = self.text_prompt(os.path.join(cfg.DATA.INDEX_LABEL_MAPPING_FILE))
        else:
            self.text_dict = self.text_prompt(os.path.join(cfg.DATA.INDEX_LABEL_MAPPING_FILE))
            """
            print("not reimplemented yet because of label text mapping")
            exit()
            self.text_dict = self.text_prompt(os.path.join(cfg.DATA.PATH_TO_DATA_DIR, 'test_openset.csv'))
            """

        self.prompt_type_num = len(self.text_dict)
        self.cls_num = self.text_dict[0].shape[0]
        self.tune_head = cfg.TUNE_HEAD
        self.dynamic_classifier = self.achieve_csf_matrix(self.text_dict, self.model)
        if self.tune_head:
            self.head = torch.nn.Parameter(self.dynamic_classifier, requires_grad=True)
        # self.test_scale = 100.
        
        # learning factor
        # if self.cfg and (self.cfg.MODEL.FINETUNE_FACTOR != 1.0 or :
        # Indicate parameters for finetuning.
        self.lr_factor = {
            "message": cfg.MODEL.FINETUNE_FACTOR,
            "stadapt": cfg.MODEL.ADAPT_FINETUNE_FACTOR,
            "mlp": cfg.MODEL.MLP_FINETUNE_FACTOR,
        }

    def _construct_network(self, cfg):
        if cfg.MODEL.ARCH == 'vitb32':
            self.model, self.preprocess = clip.load("ViT-B/32", jit=False, )
        elif cfg.MODEL.ARCH == 'vitb16':
            self.model, self.preprocess = clip.load("ViT-B/16", jit=False, )
        else:
            print("error loading arch")
            exit()
        self.model.float()    
    
    def update_state(self):
        self.dynamic_classifier = self.achieve_csf_matrix(self.text_dict, self.model)

    def forward(self, x=None, update=False):
        # shape of x(input) is (bz, channel, clip_len, h, w)
        
        assert len(x) == self.num_pathways
        x = x[0]
        bz, channel_dim, clip_len, h, w = x.shape
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(bz*clip_len, channel_dim, h, w)
        
        img_encode = self.model.encode_image(x)        
        
        if self.training:
            # img encode [bz, feat_size]
            # text_dict  {id: [400, feat_size]},
            img_encode = img_encode / img_encode.norm(dim=-1, keepdim=True)
            
            if self.tune_head:
                norm_head = self.head / self.head.norm(dim=-1, keepdim=True)
                pred = self.model.logit_scale.exp() * img_encode @ norm_head.T
            else:
                # csf_matrix = self.dynamic_classifier / self.dynamic_classifier.norm(dim=-1, keepdim=True)
                pred = self.model.logit_scale.exp() * img_encode @ self.dynamic_classifier.T
            pred = pred.reshape(bz, clip_len, -1).mean(1)

            return pred

        else:
            # img_encode [bz, feat_size]
            # dynamic_clf shape [type_num * cls_num, feat_size]
            img_encode /= img_encode.norm(dim=-1, keepdim=True)
            
            if self.tune_head:
                norm_head = self.head / self.head.norm(dim=-1, keepdim=True)
                pred = self.model.logit_scale.exp() * img_encode @ norm_head.T
            else:
                # csf_matrix = self.dynamic_classifier / self.dynamic_classifier.norm(dim=-1, keepdim=True)
                pred = self.model.logit_scale.exp() * img_encode @ self.dynamic_classifier.T
            pred = pred.reshape(bz, clip_len, -1).mean(1)
            # pred = (img_encode @ dynamic_classifier.T).view(-1, self.prompt_type_num, self.cls_num)
            # pred = self.test_scale * pred
            # pred = pred.softmax(dim=-1).mean(1)
            # pred = pred.reshape(bz, clip_len, -1).mean(1)
            return pred 
    
    def text_prompt(self, data_file):
        
        actionclip_text_aug = [f"a photo of action {{}}", f"a picture of action {{}}", f"Human action of {{}}", f"{{}}, an action",
                f"{{}} this is an action", f"{{}}, a video of action", f"Playing action of {{}}", f"{{}}",
                f"Playing a kind of action, {{}}", f"Doing a kind of action, {{}}", f"Look, the human is {{}}",
                f"Can you recognize the action of {{}}?", f"Video classification of {{}}", f"A video of {{}}",
                f"The man is {{}}", f"The woman is {{}}"]
        
        
        """ 
        text_aug = templates = [
            f'a photo of a person {{}}.',
            f'a video of a person {{}}.',
            f'a example of a person {{}}.',
            f'a demonstration of a person {{}}.',
            f'a photo of the person {{}}.',
            f'a video of the person {{}}.',
            f'a example of the person {{}}.',
            f'a demonstration of the person {{}}.',
            f'a photo of a person using {{}}.',
            f'a video of a person using {{}}.',
            f'a example of a person using {{}}.',
            f'a demonstration of a person using {{}}.',
            f'a photo of the person using {{}}.',
            f'a video of the person using {{}}.',
            f'a example of the person using {{}}.',
            f'a demonstration of the person using {{}}.',
            f'a photo of a person doing {{}}.',
            f'a video of a person doing {{}}.',
            f'a example of a person doing {{}}.',
            f'a demonstration of a person doing {{}}.',
            f'a photo of the person doing {{}}.',
            f'a video of the person doing {{}}.',
            f'a example of the person doing {{}}.',
            f'a demonstration of the person doing {{}}.',
            f'a photo of a person during {{}}.',
            f'a video of a person during {{}}.',
            f'a example of a person during {{}}.',
            f'a demonstration of a person during {{}}.',
            f'a photo of the person during {{}}.',
            f'a video of the person during {{}}.',
            f'a example of the person during {{}}.',
            f'a demonstration of the person during {{}}.',
            f'a photo of a person performing {{}}.',
            f'a video of a person performing {{}}.',
            f'a example of a person performing {{}}.',
            f'a demonstration of a person performing {{}}.',
            f'a photo of the person performing {{}}.',
            f'a video of the person performing {{}}.',
            f'a example of the person performing {{}}.',
            f'a demonstration of the person performing {{}}.',
            f'a photo of a person practicing {{}}.',
            f'a video of a person practicing {{}}.',
            f'a example of a person practicing {{}}.',
            f'a demonstration of a person practicing {{}}.',
            f'a photo of the person practicing {{}}.',
            f'a video of the person practicing {{}}.',
            f'a example of the person practicing {{}}.',
            f'a demonstration of the person practicing {{}}.',
            ]
        """
        
        text_aug = [
            f'a photo of {{}}.',
            f'a photo of a person {{}}.',
            f'a photo of a person using {{}}.',
            f'a photo of a person doing {{}}.',
            f'a photo of a person during {{}}.',
            f'a photo of a person performing {{}}.',
            f'a photo of a person practicing {{}}.',
            f'a video of {{}}.',
            f'a video of a person {{}}.',
            f'a video of a person using {{}}.',
            f'a video of a person doing {{}}.',
            f'a video of a person during {{}}.',
            f'a video of a person performing {{}}.',
            f'a video of a person practicing {{}}.',
            f'a example of {{}}.',
            f'a example of a person {{}}.',
            f'a example of a person using {{}}.',
            f'a example of a person doing {{}}.',
            f'a example of a person during {{}}.',
            f'a example of a person performing {{}}.',
            f'a example of a person practicing {{}}.',
            f'a demonstration of {{}}.',
            f'a demonstration of a person {{}}.',
            f'a demonstration of a person using {{}}.',
            f'a demonstration of a person doing {{}}.',
            f'a demonstration of a person during {{}}.',
            f'a demonstration of a person performing {{}}.',
            f'a demonstration of a person practicing {{}}.',
        ]
        
        # text_aug = text_aug + actionclip_text_aug
        # text_aug = [f"{{}}"]
        # text_aug = [f"itap of a {{}}.", f"a bad photo of the {{}}.", f"a origami {{}}.", f"a photo of thelarge {{}}.", f"a {{}} in a video game.", f"art of the {{}}.", f"a photo of the small {{}}."]

        text_dict = {}
        num_text_aug = len(text_aug)
        
        id2cls = {}
        temp_mapping = json.load(open(data_file, 'r'))
        for key in temp_mapping:
            id2cls[int(key)] = temp_mapping[key]
        """
        # parse datafile
        lines = open(data_file, 'r').readlines()
        for line in lines:
            cls_name, cls_id = line.strip().split(',')
            cls_name = cls_name.split('/')[1]
            cls_name = cls_name.replace('_', ' ')
            if cls_name not in id2cls:
                id2cls[int(cls_id)] = cls_name
        """
        cls_num = len(id2cls)
        # construct the source of dynamic classifier
        for idx, txt in enumerate(text_aug):
            text_dict[idx] = torch.cat([clip.tokenize(txt.format(id2cls[id])) for id in range(cls_num)]) 
        return text_dict
        
    def achieve_csf_matrix(self, text_dict, model):
        with torch.no_grad():
            csf_matrix_list = [model.encode_text(text_dict[i].cuda()).detach() for i in range(len(text_dict))]
            for csf_matrix in csf_matrix_list:
                csf_matrix /= csf_matrix.norm(dim=-1, keepdim=True)
        
        csf_matrix = torch.stack(csf_matrix_list, 0).mean(0)
        csf_matrix /= csf_matrix.norm(dim=-1, keepdim=True)

        # if not self.tune_head:
        #     csf_matrix *= model.logit_scale.exp()
        
        return csf_matrix
        
 

if __name__ == '__main__':
    model, preprocess = clip.load("/share/home/jia/.cache/clip/ViT-B-32.pt", jit=False, )
    
    # model: text and vision





    
