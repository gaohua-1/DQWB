import numpy as np
import torch
import math
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import init
from .pointnet_util import index_points, square_distance
class TransformerBlock(nn.Module):
    def __init__(self, d_points, d_model, k) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_points, d_model)  # 线性变换，将每个点特征从d_points维转换为d_model维
        self.fc2 = nn.Linear(d_model, d_points)  # 线性变换，将加权求和后的结果从d_model维转换为d_points维
        self.fc_delta = nn.Sequential( 
            nn.Linear(3, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.fc_gamma = nn.Sequential(  # 神经网络结构，用于学习每个点的权重,
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.w_qs = nn.Linear(d_model, d_model, bias=False)  # 线性变换，生成查询向量
        self.w_ks = nn.Linear(d_model, d_model, bias=False)  # 线性变换，生成邻居点向量
        self.w_vs = nn.Linear(d_model, d_model, bias=False)  # 线性变换，生成邻居点特征向量
        self.k = k

    def forward(self, xyz, features):
        dists = square_distance(xyz, xyz)  # 计算每两个点之间的距离，用于选取每个点最近的K个邻居点
        knn_idx = dists.argsort()[:, :, :self.k]  # 将每个点的邻居点按照距离排序，选取最近的K个邻居点
        knn_xyz = index_points(xyz, knn_idx)  # 获取每个点的K个邻居点的坐标

        pre = features  # 记录一下之前的特征向量，用于后续残差连接
        x = self.fc1(features)  # 对每个点的特征向量进行线性变换，将其从d_points维转换为d_model维
        q, k, v = self.w_qs(x), index_points(self.w_ks(x), knn_idx), index_points(self.w_vs(x),
                                                                                  knn_idx)
        # 分别生成查询向量(q)，邻居点向量(k)，邻居点特征向量(v)

        pos_enc = self.fc_delta(xyz[:, :, None] - knn_xyz)  # 学习每个邻居点与中心点之间的位置关系（相对距离），用于加权求和
        # [batch_size, n_points, n_neighbors, 3] - [batch_size, n_points, n_neighbors, 3] = [batch_size, n_points, n_neighbors, 3]，即为中心点到每个邻居点的相对距离
        # [batch_size, n_points, n_neighbors, 3] 经过self.fc_delta之后，变为 [batch_size, n_points, n_neighbors, d_model]

        attn = self.fc_gamma(q[:, :, None] - k + pos_enc)  # 计算每个邻居点的权重，加上相对位置编码
        attn = F.softmax(attn / np.sqrt(k.size(-1)), dim=-2)  # softmax归一化，保证每个中心点在所有邻居点的权重之和为1

        res = torch.einsum('bmnf,bmnf->bmf', attn, v + pos_enc)  # 加权求和，算上中心点到邻居点的相对位置编码
        # attn的shape为 [batch_size, n_points, n_neighbors, d_model]
        # v的shape为 [batch_size, n_points, n_neighbors, d_model]
        # 输出res的shape为 [batch_size, n_points, d_model]
        res = self.fc2(res) + pre  # 线性变换，将加权求和的结果从d_model维转换为d_points维，并加上残差连接
        # return res, attn
        res = res.permute(0, 2, 1)
        # print(res.shape)
        return res
class x_1d(torch.nn.Module):
    def __init__(self, feature_dim, rgb):
        super(x_1d, self).__init__()
        self.feat_dim = feature_dim
        self.rgb = rgb
        self.temporal_dim = 100
        self.c_hidden = 512
        self.output_dim = 3
        self.conv1 = torch.nn.Conv1d(in_channels=self.feat_dim//2,    out_channels=self.c_hidden,kernel_size=3,stride=1,padding=1,groups=1)
        self.conv2 = torch.nn.Conv1d(in_channels=self.c_hidden,out_channels=128,kernel_size=3,stride=1,padding=1,groups=1)
        self.conv3 = torch.nn.Conv1d(in_channels=128,out_channels=self.output_dim,   kernel_size=1,stride=1,padding=0)
        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            init.xavier_normal(m.weight)
            init.constant(m.bias, 0)

    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, x):
        if self.rgb:
            x = x[:, : self.feat_dim//2]
        else:
            x = x[:, self.feat_dim//2: ]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x4 = torch.sigmoid(0.01*self.conv3(x))
        xb_start, xb_end, xc = torch.split(x4, 1, 1)
        output_dict = {
            'xc_feat': x,
            'xc': xc,
            'xb_start': xb_start,
            'xb_end': xb_end,
        }
        return output_dict


class MCBD(nn.Module):

    def __init__(self, feature_dim, rgb):
        super(MCBD, self).__init__()
        self.hidden_dim_1d = 256
        self.hidden_dim_2d = 128
        self.hidden_dim_3d = 512
        self.num_sample = 32
        self.tscale = 128
        self.prop_boundary_ratio = 0.5
        self.num_sample_perbin = 3
        self._get_interp1d_mask()

        self.x_1d = x_1d(feature_dim, rgb)

        self.best_loss = 999999
        self.expand = 4
        self.tfblock = TransformerBlock(d_points=self.hidden_dim_2d, d_model=self.hidden_dim_2d, k=4)
        self.x_3d_p = nn.Sequential(
            nn.Conv3d(128, 512, kernel_size=(self.num_sample, 1, 1),stride=(self.num_sample, 1, 1)),
            nn.ReLU(inplace=True)
        )

        self.x_2d_p = nn.Sequential(
            nn.Conv2d(512, self.hidden_dim_2d, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hidden_dim_2d, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 4, kernel_size=1),
            nn.Sigmoid())


    def forward(self, x):
        frame_feat = self.x_1d(x)
        #print("frame_feat:",frame_feat['xc_feat'].shape)
        x = frame_feat['xc_feat']
        y = x.permute(0, 2, 1).contiguous()
        N, L, C = y.shape#特征维度 C 作为最后一维处理

        # 使用 expand 方法扩展张量
        y = y.unsqueeze(2).unsqueeze(3).expand(N, L, self.expand, self.expand, C).reshape(N, -1, C)
        # 创建三元组的索引
        indices = torch.meshgrid(
            torch.arange(L),#生成 L 个基础位置的索引
            torch.arange(self.expand),#生成 self.expand 个扩展位置的索引
            torch.arange(self.expand)#生成 self.expand 个扩展位置的索引
        )#是一个包含三个分量的元组，每个分量用于描述网格中的行、列坐标
         #将三元组索引合并为一个Tensor
        #将 indices 元组的三个分量堆叠为一个 (L, self.expand, self.expand, 3) 张量，每个位置有三维坐标 (i, j, k)。
        indices = torch.stack(indices, dim=-1).view(-1, 3)#将这个张量展平为 (L * self.expand * self.expand, 3)，其中每行表示一个位置的坐标。
        indices = indices.unsqueeze(0).expand(N, -1, -1).to('cuda').to(torch.float)

        y = self.tfblock(indices, y).view(N, C, L, self.expand * self.expand)

        y = torch.mean(y, dim=3)


        x = x + y
        confidence_map = self._boundary_matching_layer(x)
        confidence_map = self.x_3d_p(confidence_map).squeeze(2)
        prop_feat = self.x_2d_p(confidence_map)
        prop_start = prop_feat[:, 0:1].contiguous()
        prop_end = prop_feat[:, 1:2].contiguous()
        iou = prop_feat[:, 2:].contiguous()

        output_dict = {
            'xc': frame_feat['xc'],
            'xb_start': frame_feat['xb_start'],
            'xb_end': frame_feat['xb_end'],
            'iou': iou,
            'prop_start': prop_start,
            'prop_end': prop_end
        }
        return output_dict

    def _boundary_matching_layer(self, x):
        input_size = x.size()
        out = torch.matmul(x, self.sample_mask).reshape(input_size[0],input_size[1],self.num_sample,self.tscale,self.tscale)
        return out

    def _get_interp1d_bin_mask(self, seg_xmin, seg_xmax, tscale, num_sample, num_sample_perbin):
        # generate sample mask for a boundary-matching pair
        plen = float(seg_xmax - seg_xmin)
        plen_sample = plen / (num_sample * num_sample_perbin - 1.0)
        total_samples = [
            seg_xmin + plen_sample * ii
            for ii in range(num_sample * num_sample_perbin)
        ]
        p_mask = []
        for idx in range(num_sample):
            bin_samples = total_samples[idx * num_sample_perbin:(idx + 1) * num_sample_perbin]
            bin_vector = np.zeros([tscale])
            for sample in bin_samples:
                sample_upper = math.ceil(sample)
                sample_decimal, sample_down = math.modf(sample)
                if int(sample_down) <= (tscale - 1) and int(sample_down) >= 0:
                    bin_vector[int(sample_down)] += 0.5
                if int(sample_upper) <= (tscale - 1) and int(sample_upper) >= 0:
                    bin_vector[int(sample_upper)] += 0.5
            bin_vector = 1.0 / num_sample_perbin * bin_vector
            p_mask.append(bin_vector)
        p_mask = np.stack(p_mask, axis=1)
        return p_mask

    def _get_interp1d_mask(self):
        # generate sample mask for each point in Boundary-Matching Map
        mask_mat = []
        for end_index in range(self.tscale):
            mask_mat_vector = []
            for start_index in range(self.tscale):
                if start_index <= end_index:
                    p_xmin = start_index
                    p_xmax = end_index + 1
                    center_len = float(p_xmax - p_xmin) + 1
                    sample_xmin = p_xmin - center_len * self.prop_boundary_ratio
                    sample_xmax = p_xmax + center_len * self.prop_boundary_ratio
                    p_mask = self._get_interp1d_bin_mask(
                        sample_xmin, sample_xmax, self.tscale, self.num_sample,
                        self.num_sample_perbin)
                else:
                    p_mask = np.zeros([self.tscale, self.num_sample])
                mask_mat_vector.append(p_mask)
            mask_mat_vector = np.stack(mask_mat_vector, axis=2)
            mask_mat.append(mask_mat_vector)
        mask_mat = np.stack(mask_mat, axis=3)
        mask_mat = mask_mat.astype(np.float32)
        self.sample_mask = nn.Parameter(torch.Tensor(mask_mat).view(self.tscale, -1), requires_grad=False)
