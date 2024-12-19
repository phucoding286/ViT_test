import cv2
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import math


# lớp nhúng cho ViT
class ImageEmbedding(nn.Module):
    def __init__(self, 
                 embed_dim:int=512,
                 patches_size:tuple=(16, 16),
                 image_size=(32, 32, 3),
                 embed_drop:float=0.1,
                 device=None,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        self.sum_image_dim = int((image_size[0] * image_size[1] * image_size[2]))
        self.patches_num_detect = int((image_size[0] / patches_size[0]) * (image_size[1] / patches_size[1]))
        self.patches_dim_detect = int(self.sum_image_dim / self.patches_num_detect)

        self.embed_linear = nn.Linear(
            in_features=self.patches_dim_detect,
            out_features=embed_dim,
            device=device
        )
        self.dropout = nn.Dropout(embed_drop)

        self.patches_size = patches_size
        self.embed_dim = embed_dim

    def split_patches(self, image: torch.Tensor, width_cell:int=16, height_cell:int=16):
        image = image / 255
        origin_img_dim = image.size()

        wd_per_w = origin_img_dim[1] // width_cell
        wd_per_h = origin_img_dim[2] // height_cell

        patches = list()
        for i in range(wd_per_w):
            for j in range(wd_per_h):
                image_cell = image[:, i*width_cell: width_cell*(i+1), :, :]
                image_cell = image_cell[:, :, j*height_cell: height_cell*(j+1), :]
                patches.append(image_cell)
    
        return torch.stack(patches, dim=1).flatten(2)

    def pos_enc(self, max_len, d_model):
        i = torch.arange(start=0, end=d_model, step=2)
        d = torch.pow(10000.0, i/d_model)
        pos = torch.arange(start=0, end=max_len, step=1)
        pos = pos.reshape(max_len, 1)
        evenPE = torch.sin(pos/d)
        oddPE = torch.cos(pos/d)
        return torch.stack([evenPE, oddPE], dim=-1).reshape(max_len, d_model)

    def forward(self, x: torch.Tensor):
        x = self.split_patches(
            image=x,
            width_cell=self.patches_size[0],
            height_cell=self.patches_size[1]
        )
        x = self.embed_linear(x)
        return self.dropout(x + self.pos_enc(self.patches_num_detect, self.embed_dim))
    

class SelfAttentionLayer(nn.Module):
    def __init__(self,
                 d_model: int,
                 nhead: int,
                 ffn_dim: int,
                 nn_dropout: float,
                 device=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # lớp cho khối chú ý
        self.qkv_layer = nn.Linear(
            in_features=d_model,
            out_features=d_model*3,
            device=device
        )
        
        # các lớp cho khối truyền thẳng
        self.ffn_layer1 = nn.Linear(
            in_features=d_model,
            out_features=ffn_dim,
            device=device
        )
        self.ffn_h_dropout = nn.Dropout(
            p=nn_dropout
        )
        self.ffn_layer2 = nn.Linear(
            in_features=ffn_dim,
            out_features=d_model,
            device=device
        )
        self.ffn_relu_funct = nn.ReLU()
        self.ffn_o_dropout = nn.Dropout(
            p=nn_dropout
        )

        # các lớp chuẩn hóa giá trị
        self.norm1 = nn.LayerNorm(
            normalized_shape=d_model,
            device=device
        )
        self.norm2 = nn.LayerNorm(
            normalized_shape=d_model,
            device=device
        )

        self.d_model = d_model
        self.nhead = nhead
        self.ffn_dim = ffn_dim
        self.head_dim = d_model // nhead
    
    def scaled_dot_product(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, causal_mask=None):
        scaled = torch.matmul(Q, K.permute(0, 1, 3, 2)) / math.sqrt(Q.size(-1))

        if causal_mask != None:
            scaled += causal_mask

        attention_scores = F.softmax(scaled, dim=-1)
        values = torch.matmul(attention_scores, V)

        return attention_scores, values

    def att_block(self, x: torch.Tensor, causal_mask=None):
        origin_dim = x.size()

        x = self.qkv_layer(x)
        x = x.reshape(origin_dim[0], origin_dim[1], self.nhead, self.head_dim*3)
        x = x.permute(0, 2, 1, 3)
        q, k, v = torch.split(x, split_size_or_sections=self.head_dim, dim=-1)

        att_scores, values = self.scaled_dot_product(q, k, v, causal_mask)

        values = values.permute(0, 2, 1, 3)
        att_out = values.reshape(origin_dim[0], origin_dim[1], self.d_model)

        return att_scores, att_out
    
    def ffn_block(self, x: torch.Tensor):
        x = self.ffn_layer1(x)
        x = self.ffn_h_dropout(x)
        x = self.ffn_layer2(x)
        x = self.ffn_relu_funct(x)
        x = self.ffn_o_dropout(x)
        return x
    
    def forward(self, x: torch.Tensor, causal_mask=None):
        signal = x.clone()
        att_scores, att_out = self.att_block(x, causal_mask)
        x = self.norm1(att_out + signal)

        signal = x.clone()
        ffn_out = self.ffn_block(x)
        x = self.norm2(ffn_out + signal)

        return att_scores, x
    

class SelfAttCV(nn.Module):
    def __init__(self,
                 d_model:int,
                 nhead:int,
                 ffn_dim:int,
                 nn_dropout:int,
                 layers:int=1,
                 patches_size:tuple=(16, 16),
                 image_size:tuple=(32, 32, 3),
                 embed_drop:float=0.1,
                 object_size:int=1,
                 device:int=None,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        self.img_embed = ImageEmbedding(
            embed_dim=d_model,
            patches_size=patches_size,
            image_size=image_size,
            embed_drop=embed_drop,
            device=device
        )
        self.att_layers = nn.ModuleList(
            [
                SelfAttentionLayer(d_model, nhead, ffn_dim, nn_dropout, device)
                for _ in range(layers)
            ]
        )
        self.linear_out = nn.Linear(
            in_features=d_model,
            out_features=object_size+1,
            device=device
        )

        self.d_model = d_model
        self.cls_token = None
        self.device = device
    
    def forward(self, x: torch.Tensor):
        x = self.img_embed(x)
        self.cls_token = nn.Parameter(
            torch.zeros( (x.size(0), 1, self.d_model), device=self.device )
        )
        x = torch.concatenate([self.cls_token, x], dim=1)
        att_scores = None
        for layer in self.att_layers:
            att_scores, x = layer(x)
        out = self.linear_out(x[:, 0, :])
        return att_scores, out
    
    def inference_predict(self, x: torch.Tensor):
        att_scores, output = self.forward(x)
        return att_scores[:, :, 1:, 1:], torch.nn.functional.softmax(output)

image = torch.tensor(
    cv2.resize( cv2.imread("img.jpg"),(128, 64) )
).unsqueeze(0)

self_att = SelfAttCV(
    d_model=512,
    nhead=8,
    ffn_dim=1024,
    nn_dropout=0.1,
    layers=1,
    patches_size=(8, 8),
    image_size=(128, 64, 3),
    embed_drop=0.1,
    object_size=1,
    device=None
)

att_scores, output = self_att.inference_predict(image)
print(f"Mức độ chú ý cho {att_scores.size(-2)} patches:")
print(torch.sum(torch.mean(att_scores.permute(0, 2, 1, 3), dim=-1), -1))