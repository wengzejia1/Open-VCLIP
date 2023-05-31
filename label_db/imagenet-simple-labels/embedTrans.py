from transformers import AutoTokenizer, AutoModel
import json
import tqdm 
import torch

if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased")
    
    template = 'a photo of the '
    
    with open('./imagenet-simple-labels.json') as f:
        labels = json.load(f)

    def class_id_to_label(i):
            return labels[i]
    
    cls_text_embed = []
    for label in tqdm.tqdm(labels):
        text = template + label + '.'
        inputs = tokenizer(text, return_tensors="pt")
        outputs = model(**inputs)['pooler_output']
        cls_text_embed.append(outputs.squeeze(0))
    
    embed_matrix = torch.stack(cls_text_embed, 0)
    torch.save(embed_matrix, './embed_matrix.pth')
    print(embed_matrix.shape)

