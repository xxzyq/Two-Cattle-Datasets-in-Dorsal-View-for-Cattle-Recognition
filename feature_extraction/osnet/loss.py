from torch import nn
import numpy as np
from scipy.spatial.distance import cosine
from torch.nn import Parameter
import torch
import math
import torch.nn.functional as F
import itertools

class TripletLoss(nn.Module):
    def __init__(self, margin=0.5, mutual_flag=False):
        super(TripletLoss, self).__init__()
        self.margin = margin    # 三元组损失中使用的边界值（margin），控制正负样本间距的最小差值
        self.ranking_loss = nn.MarginRankingLoss(margin=margin) # 定义一个排名损失函数，这个函数用于计算triplet损失
        self.mutual = mutual_flag   #  用于控制是否启用互信息特性的标志
    def forward(self, inputs, targets):
        n = inputs.size(0)  # 获取输入的批次大小
        # 归一化输入特征，以确保它们在计算距离时具有单位范数
        inputs_norm = inputs / (torch.norm(inputs, dim=1, keepdim=True) + 1e-12)
        # 计算归一化特征之间的余弦距离
        dist = 1 - torch.matmul(inputs_norm, inputs_norm.t())
        # 创建一个掩码，标识哪些样本属于同一类
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())    
        dist_ap, dist_an = [], []# 初始化anchor-positive和anchor-negative距离列表
        for i in range(n):
            pos_mask = mask[i]   # 获取当前样本的所有正样本（相同类别）     
            neg_mask = mask[i] == 0 # 获取当前样本的所有负样本（不同类别）
            # 选择最大的正样本距离
            if dist[i][pos_mask].numel() > 0:
                dist_ap.append(dist[i][pos_mask].max().unsqueeze(0))
            else:
                # 如果没有正样本，添加无穷大以避免计算错误
                dist_ap.append(torch.tensor([float('inf')]).to(dist.device))
            # 选择最小的负样本距离
            if dist[i][neg_mask].numel() > 0:
                dist_an.append(dist[i][neg_mask].min().unsqueeze(0))
            else:
                # 如果没有负样本，同样添加无穷大
                dist_an.append(torch.tensor([float('inf')]).to(dist.device))
        # 将距离列表转换为张量
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        return loss

class PairwiseCircleLoss:
    def __init__(self, margin=0.25, gamma=512):
        """
        初始化PairwiseCircleLoss类。
        
        参数:
        margin (float): 边际值。
        gamma (float): 缩放参数。
        """
        self.margin = margin
        self.gamma = gamma

    def __call__(self, embedding: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        计算pairwise circle loss。
        
        参数:
        embedding (torch.Tensor): 样本的嵌入向量，大小为 (N, D)，其中 N 是样本数量，D 是嵌入向量的维度。
        targets (torch.Tensor): 样本的标签，大小为 (N,)。
        
        返回:
        torch.Tensor: 计算得到的损失值。
        """
        # 对嵌入向量进行L2归一化，使得计算的相似度是余弦相似度
        embedding = F.normalize(embedding, dim=1)
        # 计算嵌入向量之间的余弦相似度，得到相似度矩阵
        dist_mat = torch.matmul(embedding, embedding.t())
        # 获取样本数量
        N = dist_mat.size(0)
        # 创建正负样本掩码矩阵
        is_pos = targets.view(N, 1).expand(N, N).eq(targets.view(N, 1).expand(N, N).t()).float()
        is_neg = targets.view(N, 1).expand(N, N).ne(targets.view(N, 1).expand(N, N).t()).float()
        # 排除自身，确保正样本掩码矩阵中的对角线为0
        is_pos = is_pos - torch.eye(N, N, device=is_pos.device)
        # 计算正样本和负样本的相似度矩阵
        s_p = dist_mat * is_pos
        s_n = dist_mat * is_neg
        # 计算正样本的权重，使用detach()以避免梯度传播影响权重计算
        alpha_p = torch.clamp_min(-s_p.detach() + 1 + self.margin, min=0.)
        # 计算负样本的权重
        alpha_n = torch.clamp_min(s_n.detach() + self.margin, min=0.)
        # 定义正样本和负样本的delta值
        delta_p = 1 - self.margin
        delta_n = self.margin
        # 计算正样本的logit值，使用极小值来掩码不相关的元素
        logit_p = - self.gamma * alpha_p * (s_p - delta_p) + (-99999999.) * (1 - is_pos)
        # 计算负样本的logit值，使用极小值来掩码不相关的元素
        logit_n = self.gamma * alpha_n * (s_n - delta_n) + (-99999999.) * (1 - is_neg)
        # 计算损失值，使用softplus和logsumexp函数
        loss = F.softplus(torch.logsumexp(logit_p, dim=1) + torch.logsumexp(logit_n, dim=1)).mean()
        return loss




class GlobalTripletLoss(nn.Module):
    def __init__(self, margin=0.5):
        super(GlobalTripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.feature_store = {}  # 主特征存储（全局变量1）
        self.temp_feature_store = {}  # 临时特征存储（全局变量2）

    def update_temp_feature_store(self, inputs_norm, targets):
        # 直接添加或更新临时特征存储
        classes = torch.unique(targets)
        for class_id in classes:
            class_mask = (targets == class_id)
            class_features = inputs_norm[class_mask]
            class_feature_mean = class_features.mean(0)
            self.temp_feature_store[class_id.item()] = class_feature_mean.detach()

    def merge_feature_stores(self):
        # 结束时，完全覆盖主特征存储
        self.feature_store = self.temp_feature_store.copy()
        self.temp_feature_store.clear()  # 重置临时特征存储
        

    def forward(self, inputs, targets):
        n = inputs.size(0)
        inputs_norm = inputs / (torch.norm(inputs, dim=1, keepdim=True) + 1e-12)
        self.update_temp_feature_store(inputs_norm, targets)

        dist_ap, dist_an = [], []
        for i in range(n):
            pos_mask = targets == targets[i]
            neg_mask = targets != targets[i]

            # Positive samples distance
            dist_ap.append((inputs_norm[i] - inputs_norm[pos_mask]).pow(2).sum(1).max().unsqueeze(0))
            # Choose the hardest negative samples from feature_store if available
            if self.feature_store:
                neg_features = torch.stack([self.feature_store[tid.item()] for tid in targets[neg_mask]])
            else:
                neg_features = inputs_norm[neg_mask]

            dist_an.append((inputs_norm[i] - neg_features).pow(2).sum(1).min().unsqueeze(0))

        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        return loss
    
# class ArcFaceLoss(nn.Module):
#     def __init__(self, num_classes, emb_size, s=120.0, m=0.3, easy_margin=False):
#         super(ArcFaceLoss, self).__init__()
#         # self.num_classes = num_classes # 输入特征的维度
#         # self.emb_size = emb_size    # 输出特征（类别）的维度
#         self.weight = Parameter(torch.FloatTensor(num_classes, emb_size)).to('cuda:0')
#         # num_classes 训练集中总的人脸分类数
#         # emb_size 特征向量长度
#         # nn.init.xavier_uniform_(self.weight)
#         self.initialize_weights(num_classes, emb_size)
        
#         # nn.init.xavier_uniform_(self.weight)
#         # with open('W.txt', 'w') as f:
#         #     f.write(f'{self.weight}\n')
#         with open("W.txt", 'w') as f:
#             for feat in self.weight:
#                 feat_str = ' '.join(map(str, feat.tolist()))
#                 f.write(feat_str + '\n')
#         # 使用均匀分布来初始化weight
#         self.easy_margin = easy_margin
#         self.m = m
#         # 夹角差值 0.5 公式中的m
#         self.s = s
#         # 半径 64 公式中的s
#         # 二者大小都是论文中推荐值
#         self.cos_m = math.cos(self.m)
#         self.sin_m = math.sin(self.m)
#         # 差值的cos和sin
#         self.th = math.cos(math.pi - self.m)
#         # 阈值，避免theta + m >= pi
#         self.mm = math.sin(math.pi - self.m) * self.m
        
#     def initialize_weights(self, num_classes, emb_size):
#             # Calculate the number of combinations of 2 positions out of emb_size
#             num_combinations = emb_size * (emb_size - 1) // 2
#             assert num_combinations >= num_classes, "Not enough combinations to ensure orthogonality with given emb_size."
#             # Create all possible combinations of 2 ones in emb_size length vector
#             all_combinations = list(itertools.combinations(range(emb_size), 2))
#             # Randomly select num_classes combinations
#             selected_combinations = all_combinations[:num_classes]
#             # Create weight matrix
#             weight_matrix = torch.zeros((num_classes, emb_size))
#             for i, comb in enumerate(selected_combinations):
#                 weight_matrix[i, comb[0]] = 1
#                 weight_matrix[i, comb[1]] = 1
#             self.weight.data = weight_matrix.to('cuda:0')

#     def forward(self, input, label):
#         x = F.normalize(input)
#         W = F.normalize(self.weight)
#         # print(W)
#         # print(x.shape,W.shape)
#         # 正则化
#         cosine = F.linear(x, W)  # (batchsize, num_classes)
#         # cos值
#         sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
#         # sin
#         phi = cosine * self.cos_m - sine * self.sin_m
#         # print(W.shape)
#         # cos(theta + m) 余弦公式
#         if self.easy_margin:
#             phi = torch.where(cosine > 0, phi, cosine)
#             # 如果使用easy_margin
#         else:
#             phi = torch.where(cosine > self.th, phi, cosine - self.mm)
#         one_hot = torch.zeros(cosine.size(), device='cuda')
#         one_hot.scatter_(1, label.view(-1, 1).long(), 1)
#         # 将样本的标签映射为one hot形式 例如N个标签，映射为（N，num_classes）
#         output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
#         # 对于正确类别（1*phi）即公式中的cos(theta + m)，对于错误的类别（1*cosine）即公式中的cos(theta）
#         # 这样对于每一个样本，比如[0,0,0,1,0,0]属于第四类，则最终结果为[cosine, cosine, cosine, phi, cosine, cosine]
#         # 再乘以半径，经过交叉熵，正好是ArcFace的公式
#         output *= self.s
#         # 乘以半径
#         return F.cross_entropy(output, label)   # 计算并返回交叉熵损失

class ArcFaceLoss(nn.Module):
    def __init__(self, num_classes, emb_size, s=120.0, m=0.3, easy_margin=False):
        super(ArcFaceLoss, self).__init__()
        self.weight = Parameter(torch.FloatTensor(num_classes, emb_size)).to('cuda:0')

        nn.init.xavier_uniform_(self.weight)
        self.initialize_weights(num_classes, emb_size,self.weight)
        
        with open("W.txt", 'w') as f:
            for feat in self.weight:
                feat_str = ' '.join(map(str, feat.tolist()))
                f.write(feat_str + '\n')
                
        self.easy_margin = easy_margin
        self.m = m
        self.s = s
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m

    def initialize_weights(self, num_classes, emb_size, weight):
        
        # 使用新的QR分解方法来生成正交向量
        Q, _ = torch.linalg.qr(weight, mode='reduced')
        
        # 确保零值比例
        zero_percentage = 0.5
        zero_count = int(zero_percentage * emb_size)
        for i in range(num_classes):
            indices = torch.randperm(emb_size)[:zero_count]
            Q[i, indices] = 0
        
        self.weight.data = Q.to('cuda:0')

    def forward(self, input, label):
        x = F.normalize(input)
        W = F.normalize(self.weight)
        
        cosine = F.linear(x, W)  # (batchsize, num_classes)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        
        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        
        return F.cross_entropy(output, label)