import torch
import torch.nn as nn
class Stem(nn.Module):

    def __init__(self, input_dim=3, output_dim=None, patch_size=32): # 32
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=patch_size, stride=patch_size), # 8
            nn.BatchNorm2d(output_dim),
        )
    def forward(self, x):
        B, T, P, C, H, W = x.shape
        x = x.view(-1, C, H, W)
        x = self.stem(x) # BxTxP, C, 1, 1
        x = x.view(B, T, P, x.shape[1]) # B, T, P, C
        return x
class Grapher(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
        )
        self.fc2 = nn.Sequential(
            nn.Conv2d(out_channels*2, in_channels, 1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )

        self.fc3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
        )

        self.fc4 = nn.Sequential(
            nn.Conv2d(out_channels*2, in_channels, 1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )
        self.InterPartMR = InterPartMR(out_channels)
        self.IntraPartMR = IntraPartMR(out_channels)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
    def forward(self, x):
        # B, T, P, C = x.shape
        B, T, C, P, _ = x.shape # B,T,C,P,1
        x = x.view(-1,C,P,1) # BxT, C, P, 1
        tmp_x = x
        x = self.fc1(x)
        x = self.InterPartMR(x) # BxT, C*5, P, 1
        x = self.fc2(x)
        x = x+tmp_x
        x = self.act(x)
        x = self.fc3(x)
        x = self.IntraPartMR(x)

        x = self.fc4(x)
        x = x + tmp_x
        x = self.act(x)
        return x.view(B,T,C,P,1)


class InterPartMR(nn.Module):
    def __init__(self, out_channels):
        super().__init__()
        self.nn = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels * 2, 1, groups=4),
            nn.BatchNorm2d(out_channels * 2),
            nn.ReLU()
        )

        # 分部索引定义（6类）
        self.part_indices = [
            [0, 1, 2, 3, 4],  # 0: 头部
            [5, 7, 9],  # 1: 左肩
            [6, 8, 10],  # 2: 右肩
            [11, 13, 15],  # 3: 左腿
            [12, 14, 16],  # 4: 右腿
            [17, 18, 19, 20],  # 5: 物体（obj）
        ]

    def forward(self, x):
        B, C, P, _ = x.shape  # BxT, C, P, 1
        tmp_x = x.clone()

        # 构建 x_i 和 x_j
        x_i = x.repeat(1, 1, 1, P)  # [B, C, P, P]
        x_j = torch.zeros_like(x_i)

        for k in range(P):
            x_j[:, :, :, k] = x[:, :, k, 0].unsqueeze(-1).repeat(1, 1, P)

        relative = x_j - x_i

        # 分部处理
        for indices in self.part_indices:
            tmp_relative = relative.clone()
            tmp_relative[:, :, :, indices] -= 1e4  # 屏蔽当前部分
            tmp_x_j, _ = torch.max(tmp_relative, -1, keepdim=True)
            tmp_x[:, :, indices, :] = tmp_x_j[:, :, indices, :]

        x = torch.cat([x, tmp_x], dim=1)  # 拼接原始和更新后的特征
        return self.nn(x)


class IntraPartMR(nn.Module):
    def __init__(self, out_channels):
        super().__init__()
        self.nn = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels * 2, 1, groups=4),
            nn.BatchNorm2d(out_channels * 2),
            nn.ReLU()
        )

        # 定义6个身体部分索引
        self.part_indices = [
            [0, 1, 2, 3, 4],  # 头部
            [5, 7, 9],  # 左肩
            [6, 8, 10],  # 右肩
            [11, 13, 15],  # 左腿
            [12, 14, 16],  # 右腿
            [17, 18, 19, 20]  # obj
        ]

        # 创建 point_id -> part_id 的映射表
        self.point_to_part = {}
        for part_id, indices in enumerate(self.part_indices):
            for idx in indices:
                self.point_to_part[idx] = part_id

    def forward(self, x):
        B, C, P, _ = x.shape  # [B, C, 21, 1]
        tmp_x = x.clone()

        # 构建 x_i 和 x_j
        x_i = x.repeat(1, 1, 1, P)  # [B, C, P, P]
        x_j = torch.zeros_like(x_i)
        for k in range(P):
            x_j[:, :, :, k] = x[:, :, k, 0].unsqueeze(-1).repeat(1, 1, P)

        relative = x_j - x_i  # [B, C, P, P]

        # 每个关键点只在自己所属 part 内计算 max
        for point in range(P):
            part_id = self.point_to_part[point]
            indices = self.part_indices[part_id]

            # 对这个点在其 part 范围内做最大值计算
            tmp_x_j, _ = torch.max(relative[:, :, point, indices], dim=-1, keepdim=True)
            tmp_x[:, :, point, :] = tmp_x_j

        x = torch.cat([x, tmp_x], dim=1)  # [B, 2C, P, 1]
        return self.nn(x)