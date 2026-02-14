"""
SimCLR Encoder model definition — extracted from train/train_oasis.py.
This avoids importing train_oasis.py which sets GPU environment variables at module level.
"""

import torch
import torch.nn as nn


class SimCLREncoder(nn.Module):
    def __init__(self, k=20, emb_dim=512):
        super(SimCLREncoder, self).__init__()
        self.k = k
        self.emb_dim = emb_dim

        self.conv1 = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64), nn.LeakyReLU(0.2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64), nn.LeakyReLU(0.2))
        self.conv3 = nn.Sequential(
            nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128), nn.LeakyReLU(0.2))
        self.conv4 = nn.Sequential(
            nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256), nn.LeakyReLU(0.2))

        self.conv5 = nn.Sequential(
            nn.Conv1d(512, emb_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(emb_dim), nn.LeakyReLU(0.2))

        self.projection_head = nn.Sequential(
            nn.Linear(emb_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128)
        )

    def knn(self, x, k):
        inner = -2 * torch.matmul(x.transpose(2, 1), x)
        xx = torch.sum(x ** 2, dim=1, keepdim=True)
        pairwise_distance = -xx - inner - xx.transpose(2, 1)
        idx = pairwise_distance.topk(k=k, dim=-1)[1]
        return idx

    def get_graph_feature(self, x, k=20, idx=None):
        batch_size = x.size(0)
        num_points = x.size(2)
        x = x.view(batch_size, -1, num_points)
        if idx is None:
            idx = self.knn(x, k=k)
        device = x.device
        idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        _, num_dims, _ = x.size()
        x = x.transpose(2, 1).contiguous()
        feature = x.view(batch_size * num_points, -1)[idx, :]
        feature = feature.view(batch_size, num_points, k, num_dims)
        x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
        feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()
        return feature

    def backbone(self, x):
        if x.shape[2] == 3:
            x = x.permute(0, 2, 1)

        x_f = self.get_graph_feature(x, k=self.k)
        x = self.conv1(x_f)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x_f = self.get_graph_feature(x1, k=self.k)
        x = self.conv2(x_f)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x_f = self.get_graph_feature(x2, k=self.k)
        x = self.conv3(x_f)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x_f = self.get_graph_feature(x3, k=self.k)
        x = self.conv4(x_f)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.conv5(x)
        x = torch.max(x, 2)[0]
        return x

    def forward(self, x):
        h = self.backbone(x)
        z = self.projection_head(h)
        return z
