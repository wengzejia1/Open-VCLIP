import torch
import torch.nn as nn
import sys
from . import clip
from .build import MODEL_REGISTRY

# from timm.models.registry import register_model

import os
import numpy as np
import json

@MODEL_REGISTRY.register()
class ClipImage(nn.Module):
    
    def __init__(self, cfg):
        """
        The `__init__` method of any subclass should also contain these
            arguments.
        Args:
            nothing.
        """
        super(ClipImage, self).__init__()
        self.cfg = cfg
        self._construct_network(cfg)
        
        # text encoder
        self.model.eval()
        assert self.cfg.IMAGENET_SIMPLELABEL_PATH != 'None'
        # self.text_dict = self.text_prompt(self.cfg.IMAGENET_SIMPLELABEL_PATH)
        self.text_dict = self.text_prompt(cfg.DATA.INDEX_LABEL_MAPPING_FILE)
        self.prompt_type_num = len(self.text_dict)
        self.cls_num = self.text_dict[0].shape[0]
        self.dynamic_classifier = self.achieve_csf_matrix(self.text_dict, self.model)
        self.test_scale = 1.

        # self.linear = torch.nn.Linear(512, 1000)

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
        
        x = x[0]
        # shape of x(input) is (bz, channel, h, w)
        bz, channel_dim, h, w = x.shape
        img_encode = self.model.encode_image(x)
        img_encode = img_encode / img_encode.norm(dim=-1, keepdim=True)
        csf_matrix = self.dynamic_classifier / self.dynamic_classifier.norm(dim=-1, keepdim=True)

        pred = self.model.logit_scale.exp() * (img_encode @ csf_matrix.T)
        return pred
        
    def text_prompt(self, data_file):
        text_aug = [f"itap of a {{}}.", f"a bad photo of the {{}}.", f"a origami {{}}.", f"a photo of the large {{}}.", f"a {{}} in a video game.", f"art of the {{}}.", f"a photo of the small {{}}."]
        """
        text_aug = imagenet_templates = [
	    f'a bad photo of a {{}}.',
	    f'a photo of many {{}}.',
	    f'a sculpture of a {{}}.',
	    f'a photo of the hard to see {{}}.',
	    f'a low resolution photo of the {{}}.',
	    f'a rendering of a {{}}.',
	    f'graffiti of a {{}}.',
	    f'a bad photo of the {{}}.',
	    f'a cropped photo of the {{}}.',
	    f'a tattoo of a {{}}.',
	    f'the embroidered {{}}.',
	    f'a photo of a hard to see {{}}.',
	    f'a bright photo of a {{}}.',
	    f'a photo of a clean {{}}.',
	    f'a photo of a dirty {{}}.',
	    f'a dark photo of the {{}}.',
	    f'a drawing of a {{}}.',
	    f'a photo of my {{}}.',
	    f'the plastic {{}}.',
	    f'a photo of the cool {{}}.',
	    f'a close-up photo of a {{}}.',
	    f'a black and white photo of the {{}}.',
	    f'a painting of the {{}}.',
	    f'a painting of a {{}}.',
	    f'a pixelated photo of the {{}}.',
	    f'a sculpture of the {{}}.',
	    f'a bright photo of the {{}}.',
	    f'a cropped photo of a {{}}.',
	    f'a plastic {{}}.',
	    f'a photo of the dirty {{}}.',
	    f'a jpeg corrupted photo of a {{}}.',
	    f'a blurry photo of the {{}}.',
	    f'a photo of the {{}}.',
	    f'a good photo of the {{}}.',
	    f'a rendering of the {{}}.',
	    f'a {{}} in a video game.',
	    f'a photo of one {{}}.',
	    f'a doodle of a {{}}.',
	    f'a close-up photo of the {{}}.',
	    f'a photo of a {{}}.',
	    f'the origami {{}}.',
	    f'the {{}} in a video game.',
	    f'a sketch of a {{}}.',
	    f'a doodle of the {{}}.',
	    f'a origami {{}}.',
	    f'a low resolution photo of a {{}}.',
	    f'the toy {{}}.',
	    f'a rendition of the {{}}.',
	    f'a photo of the clean {{}}.',
	    f'a photo of a large {{}}.',
	    f'a rendition of a {{}}.',
	    f'a photo of a nice {{}}.',
	    f'a photo of a weird {{}}.',
	    f'a blurry photo of a {{}}.',
	    f'a cartoon {{}}.',
	    f'art of a {{}}.',
	    f'a sketch of the {{}}.',
	    f'a embroidered {{}}.',
	    f'a pixelated photo of a {{}}.',
	    f'itap of the {{}}.',
	    f'a jpeg corrupted photo of the {{}}.',
	    f'a good photo of a {{}}.',
	    f'a plushie {{}}.',
	    f'a photo of the nice {{}}.',
	    f'a photo of the small {{}}.',
	    f'a photo of the weird {{}}.',
	    f'the cartoon {{}}.',
	    f'art of the {{}}.',
	    f'a drawing of the {{}}.',
	    f'a photo of the large {{}}.',
	    f'a black and white photo of a {{}}.',
	    f'the plushie {{}}.',
	    f'a dark photo of a {{}}.',
	    f'itap of a {{}}.',
	    f'graffiti of the {{}}.',
	    f'a toy {{}}.',
	    f'itap of my {{}}.',
	    f'a photo of a cool {{}}.',
	    f'a photo of a small {{}}.',
	    f'a tattoo of the {{}}.',
	]
        """

        text_dict = {}
        id2cls = {}
        num_text_aug = len(text_aug)
        temp_mapping = json.load(open(data_file, 'r'))
        for key in temp_mapping:
            id2cls[int(key)] = temp_mapping[key]
        """
        # parse datafile
        cls_list = json.load(open(data_file, 'r'))

        for idx, cls_name in enumerate(cls_list):
            id2cls[idx] = cls_name
        cls_num = len(id2cls)
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

        return csf_matrix 


if __name__ == '__main__':
    model, preprocess = clip.load("/share/home/jia/.cache/clip/ViT-B-16.pt", jit=False, )
    
   
