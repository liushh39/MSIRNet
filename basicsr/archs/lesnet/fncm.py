#!/usr/bin/env python
# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from einops import rearrange
import math

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class FRNet(nn.Module):
    """
    Feature refinement network：
    (1) IEU
    (2) CSGate
    """

    def __init__(self, channels, height=None, width=None, weight_type="bit", num_layers=1, att_size=8, mlp_layer=256):
        """
        :param channels: number of channels in the image
        :param height: height of the image
        :param width: width of the image
        type: bit or vector
        """
        super(FRNet, self).__init__()
        # IEU_G computes complementary features.
        self.IEU_G = IEU(channels, height, width, weight_type="bit",
                         bit_layers=num_layers, att_size=att_size, mlp_layer=mlp_layer)

        # IEU_W computes bit-level or vector-level weights.
        self.IEU_W = IEU(channels, height, width, weight_type=weight_type,
                         bit_layers=num_layers, att_size=att_size, mlp_layer=mlp_layer)

    def forward(self, x_img):
        com_feature = self.IEU_G(x_img)
        wegiht_matrix = torch.sigmoid(self.IEU_W(x_img))
        # CSGate
        x_out = x_img * wegiht_matrix + com_feature * (torch.tensor(1.0) - wegiht_matrix)
        return x_out

class FRNet_modified(nn.Module):
    """
    Feature refinement network：
    (1) IEU
    (2) CSGate
    """

    def __init__(self, channels, height=None, width=None, weight_type="bit", num_layers=1, att_size=8, mlp_layer=256):
        """
        :param channels: number of channels in the image
        :param height: height of the image
        :param width: width of the image
        type: bit or vector
        """
        super(FRNet_modified, self).__init__()
        # IEU_G computes complementary features.
        self.IEU_G = IEU(channels, height, width, weight_type="bit",
                         bit_layers=num_layers, att_size=att_size, mlp_layer=mlp_layer)

        # IEU_W computes bit-level or vector-level weights.
        self.IEU_W = IEU(channels, height, width, weight_type=weight_type,
                         bit_layers=num_layers, att_size=att_size, mlp_layer=mlp_layer)

    def forward(self, x_img, x_img1, x_img2):
        # com_feature = self.IEU_G(x_img)
        wegiht_matrix = torch.sigmoid(self.IEU_W(x_img))
        # CSGate
        x_out = x_img1 * wegiht_matrix + x_img2 * (torch.tensor(1.0) - wegiht_matrix)
        return x_out


class IEU(nn.Module):
    """
    Information extraction Unit (IEU) for FRNet
    (1) Self-attention
    (2) DNN
    """

    def __init__(self, channels, height=None, width=None, weight_type="bit",
                 bit_layers=1, att_size=20, mlp_layer=256):
        """
        :param channels: number of channels in the image
        :param height: height of the image
        :param width: width of the image
        :param weight_type: type of weight computation (bit or vector)
        :param bit_layers: number of layers in MLP for bit-level weight computation
        :param att_size: size of attention layer for self-attention computation
        :param mlp_layer: size of MLP layer for contextual information extraction
        """
        super(IEU, self).__init__()
        # self.input_dim = channels * height * width
        self.weight_type = weight_type

        # Self-attention unit, which is used to capture cross-feature relationships.

        # self.vector_info = SelfAttentionIEU(embed_dim=channels, att_size=att_size)
        self.vector_info = multi_head_SelfAttentionIEU(embed_dim=channels, att_size=att_size)

        # Contextual information extractor (CIE), we adopt MLP to encode contextual information.
        # mlp_layers = [mlp_layer for _ in range(bit_layers)]
        # self.mlps = MultiLayerPerceptronPrelu(self.input_dim, embed_dims=mlp_layers,
        #                                       output_layer=False)
        # self.bit_projection = nn.Linear(mlp_layer, height*width)

        self.activation = nn.ReLU()
        self.bit_projection_mlp = nn.Linear(channels, 1)

        self.Mlp = Mlp(in_features=channels, hidden_features=channels*2, act_layer=nn.GELU)

    def forward(self, x_img):
        """
        :param x_img: input image tensor (1*256*32*32),b*c*h*w
        :return: tensor with bit-level weights or complementary features (b*c*h*w)
                 or tensor with vector-level weights (b*1*h*w)
        """
        B, C, H, W = x_img.shape

        # （1）self-attetnion unit
        x_vector = self.vector_info(x_img)  # B,C,H,W

        # (2) CIE unit

        # x_bit = self.mlps(x_img.view(B, C*H*W))
        # x_bit = self.bit_projection(x_bit).view(B, H, W).unsqueeze(1)  # B,1,H,W
        # x_bit = self.activation(x_bit)

        # using the defined MultiLayerPerceptronPrelu
        x_bit = self.Mlp(x_img)
        x_bit = self.bit_projection_mlp(x_bit)
        x_bit = x_bit.permute(0, 2, 1).view(B, 1, H, W)

        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.MLPP = MLP(C*H*W, C, C*H*W).to(device)
        # x_bit = self.MLPP(x_img)
        # x_bit = self.bit_projection_mlp(x_bit.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        x_bit = self.activation(x_bit)

        # (3) Integration unit
        x_out = x_bit * x_vector

        if self.weight_type == "vector":
            # To compute vector-level importance in IEU_W
            x_out = torch.sum(x_out, dim=2, keepdim=True)
            # 1,1,H,W
            return x_out

        return x_out


####________ 2-dimentional SelfAttentionIEU
# class SelfAttentionIEU(nn.Module):
#     def __init__(self, embed_dim, att_size=20):
#         """
#         :param embed_dim:
#         :param att_size:
#         """
#         super(SelfAttentionIEU, self).__init__()
#         self.embed_dim = embed_dim
#         self.trans_Q = nn.Linear(embed_dim, att_size)
#         self.trans_K = nn.Linear(embed_dim, att_size)
#         self.trans_V = nn.Linear(embed_dim, att_size)
#         self.projection = nn.Linear(att_size, embed_dim)
#
#     def forward(self, x, scale=None):
#         """
#         :param x: B,C,H,W
#         :return: B,C,H,W
#         """
#         B, C, H, W = x.shape
#         x = x.permute(0, 2, 3, 1).view(B, -1, self.embed_dim)   # B,H*W,C
#         Q = self.trans_Q(x)
#         K = self.trans_K(x)
#         V = self.trans_V(x)
#         attention = torch.matmul(Q, K.permute(0, 2, 1))  # B,F,F
#         attention_score = F.softmax(attention, dim=-1)
#         context = torch.matmul(attention_score, V)
#         # Projection
#         context = self.projection(context)
#         context = context.permute(0, 2, 1).view(B, C, H, W)
#         return context


####________ 3-dimentional SelfAttentionIEU
class SelfAttentionIEU(nn.Module):
    def __init__(self, embed_dim, att_size=20):
        """
        :param embed_dim:
        :param att_size:
        """
        super(SelfAttentionIEU, self).__init__()
        self.embed_dim = embed_dim
        self.trans_Q = nn.Linear(embed_dim, att_size)
        self.trans_K = nn.Linear(embed_dim, att_size)
        self.trans_V = nn.Linear(embed_dim, att_size)
        self.projection = nn.Linear(att_size, embed_dim)

    def forward(self, x, scale=None):
        """
        :param x: B,C,H,W
        :return: B,C,H,W
        """
        B, C, H, W = x.shape   # B,C,H,W
        x = x.permute(0, 2, 3, 1)   # B,H,W,C
        Q = self.trans_Q(x)
        K = self.trans_K(x)
        V = self.trans_V(x)
        attention = torch.einsum('b h i d, b h j d -> b h i j', Q, K)
        attention_score = F.softmax(attention, dim=-1)
        context = torch.einsum('b h i j, b h j d -> b h i d', attention_score, V)
        # Projection
        context = self.projection(context)
        context = context.permute(0, 3, 1, 2)
        return context

####________ 3-dimentional multi-windows multi-head SelfAttentionIEU
class multi_head_SelfAttentionIEU(nn.Module):
    def __init__(self, embed_dim, att_size=20, drop=0.):
        """
        :param embed_dim:
        :param att_size:
        """
        super(multi_head_SelfAttentionIEU, self).__init__()
        self.embed_dim = embed_dim
        self.to_qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.heads = embed_dim // (embed_dim//4)
        self.drop = nn.Dropout(drop)
        self.scale = (embed_dim//4) ** -0.5

    def forward(self, x, window_size=8):
        """
        :param x: B,C,H,W
        :return: B,C,H,W
        """
        B, C, H, W = x.shape   # B,C,H,W
        x = rearrange(x, 'b d (x w1) (y w2) -> b x y w1 w2 d', w1 = window_size, w2 = window_size)    # rearrange vector
        batch, height, width, window_height, window_width, _ = x.shape
        x = rearrange(x, 'b x y w1 w2 d -> (b x y) (w1 w2) d')                        # flatten
        Q, K, V = self.to_qkv(x).chunk(3, dim=-1)                                     # project for queries, keys, values
        Q, K, V = map(lambda t: rearrange(t, 'b n (h d ) -> b h n d', h=self.heads), (Q, K, V))
        Q = Q * self.scale

        attention = torch.einsum('b h i d, b h j d -> b h i j', Q, K)
        attention_score = F.softmax(attention, dim=-1)
        context = torch.einsum('b h i j, b h j d -> b h i d', attention_score, V)

        out = rearrange(context, 'b h (w1 w2) d -> b w1 w2 (h d)', w1=window_height, w2=window_width)   # merge heads
        out = self.drop(out)
        out = rearrange(out, '(b x y) ... -> b x y ...', x=height, y=width)
        out = rearrange(out, 'b x y w1 w2 d -> b d (x w1) (y w2)')

        return out




####________orignial MLP
class MultiLayerPerceptronPrelu(torch.nn.Module):
    def __init__(self, input_dim, embed_dims, dropout=0.5, output_layer=True):
        super().__init__()
        layers = list()
        for embed_dim in embed_dims:
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            layers.append(torch.nn.BatchNorm1d(embed_dim))
            layers.append(torch.nn.PReLU())
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = embed_dim

        if output_layer:
            layers.append(torch.nn.Linear(input_dim, 1))
        self.mlp = torch.nn.Sequential(*layers)
        self._init_weight_()

    def _init_weight_(self):
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        """
        :param x: input tensor (B,C*H*W)
        :return: output tensor after MLP computation (B,)
        """
        return self.mlp(x)


####________ 1-dimentional orignial MLP
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, drop=0.):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.inorm = nn.InstanceNorm1d(hidden_dim)
        self.PReLU = nn.PReLU()
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        """
        :param x: input tensor (B,C,H,W)
        :return: output tensor after MLP computation (B,C,H,W)
        """
        B, C, H, W = x.shape
        x = x.view(B, C*H*W)
        x = self.fc1(x)
        x = self.inorm(x.view(B, -1, self.fc1.out_features)).view(B, -1)
        x = self.PReLU(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = x.view(B, C, H, W)
        return x

####________ 2-dimentional orignial MLP
class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.hidden_features = hidden_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.norm = nn.InstanceNorm2d(hidden_features)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.view(B, C, -1).permute(0, 2, 1)
        x = self.fc1(x)

        x = x.permute(0, 2, 1).view(B, -1, H, W)
        x = self.norm(x)
        x = x.view(B, self.hidden_features, -1).permute(0, 2, 1)

        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        # x = self.drop(x)
        return x



if __name__ == '__main__':
    img_tensor = torch.randn(1, 256, 32, 32)

    frnet = FRNet(channels=256, height=32, width=32)

    output_tensor = frnet(img_tensor)

    print(output_tensor.size())

