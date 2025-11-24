import numpy as np
import matplotlib.pyplot as plt
from utils import Calculating
from risk_calculating import *
import torch
from torch.func import vmap
import torch.nn as nn
def norm_data(self, da):
    return (da - da.min()) / (da.max() - da.min())

class Mapping_base:
    def __init__(self, args):
        """
        一个点，扩散的形状和速度（函数控制），扩散多远，重叠方式，初始权重,
        误报率的算法是重叠部分的均值是否大于经验阀值
        """
        self.args = args
        self.device = args.device
        self.cal = Calculating()
        self.camp = ["viridis", "RdYlGn_r", "Greens", "Reds",  'Purples']
        self.grid = args.grid_mapping
        self.ext = args.extent
        self.initial()

    def initial(self):
        lon_min, lon_max, lat_min, lat_max = self.ext
        lon_grid = torch.linspace(lon_min, lon_max, self.grid["lon"]).to(self.args.device)
        lat_grid = torch.linspace(lat_min, lat_max, self.grid["lat"]).to(self.args.device)
        self.Lon, self.Lat = torch.meshgrid(lon_grid, lat_grid, indexing='xy')
        self.map = torch.stack((self.Lon, self.Lat), axis=-1).reshape(-1, 2)
        self.map_risk = torch.zeros_like(self.map)[:, 0].to(self.args.device)

    def soomth_mtras(self, tras):
        k = 7
        x = tras[..., :2]
        xp = torch.cat([x[:1].expand(k // 2, -1, -1, -1), x, x[-1:].expand(k // 2, -1, -1, -1)], dim=0)
        cs = torch.cat([torch.zeros_like(xp[:1]), xp.cumsum(dim=0)], dim=0)
        tras_ = tras.clone()
        tras_[..., :2] = (cs[k:] - cs[:-k]) / k
        return tras_
    def show_pics(self, map_risk, pred):

        map_risk, pred = map_risk.numpy(), pred.numpy()
        plt.figure(figsize=(16, 14))
        map_risk = map_risk.reshape(self.grid["lat"], self.grid["lon"])
        plt.contourf(self.Lon.numpy(), self.Lat.numpy(), map_risk , levels=20, cmap='Reds', alpha=0.85, zorder=0)

        #plt.plot(real[:, :, 0], real[:, :, 1], color="blue", lw=1, zorder=1)
        plt.plot(pred[:, :, 0], pred[:, :, 1], color="green", lw=1, zorder=0)
        #plt.scatter(self.Lon.numpy(), self.Lat.numpy(), c="b", alpha=0.1, s=5, zorder=1)

        imp = plt.imread("/Users/yangkaisen/MyProject/Data/map/hongkong2.jpg")
        plt.imshow(imp, extent=[114.099003, 114.187537, 22.265695, 22.322062], zorder=3)
        #plt.savefig(f"./test_risk2.jpg", dpi=300, bbox_inches='tight')
        plt.show()
        print("saved")

    def forward_time(self, tras):
        return

    def forward_batch(self, tras):
        return


    def forward (self, tras):
        acc_risk = self.forward_batch(tras)
        if self.args.pred_type == "multi_pred":
            tras = self.soomth_mtras(tras)
            tras = tras.view(tras.shape[0], -1, tras.shape[-1])
        self.show_pics(self.map_risk, tras)
        return acc_risk


class Mapping_CRI(Mapping_base):
    def __init__(self, args):
        super().__init__(args)
        self.device = args.device
        self.risk_cal = CRI_Cal(args)

    def forward_time(self, tras):

        if not self.args.pred_type == "multi_pred":
            T, B, F = tras.shape
            risks = self.risk_cal.forward(tras)

            radius_batched = vmap(lambda x: self.cal.adj_dis(x, x))
            radius = radius_batched(tras[:,:,:2])
            radius = radius * (risks > 0).float()

            radius = radius.amax(dim=-1).unsqueeze(-1)
            risks = risks.mean(dim=-1).unsqueeze(-1)
            tras = tras[..., :2]
        else:
            T, B, M, F = tras.shape
            # cal modes risks [T, B, B, M]
            risks = [self.risk_cal.forward(tras[:,:,m]) for m in range(self.args.K)]
            risks = torch.stack(risks, dim=-1)

            # cal diffusion radius for each modes [T, B, B, M]
            radius_batched = vmap(lambda x: self.cal.adj_dis(x, x))
            radius = [radius_batched(tras[:,:,m, :2]) for m in range(self.args.K)]
            radius = torch.stack(radius, dim=-1)

            # keep diffusion radius that is risky edges, and summarize infos to the points
            radius = (radius * (risks > 0).float()).amax(dim=2)
            risks = risks.mean(dim=2)

            tras = tras[..., :2].view(T, -1, 2)
            risks, radius = risks.view(T, -1, 1), radius.view(T, -1, 1)

        acc_risk = torch.zeros(T, device=self.device)

        for t in range(T):
            field = self.cal.adj_dis(tras[t], self.map)
            sigma = (radius[t] / torch.sqrt(-2 * torch.log(torch.tensor(0.01, device=self.device))))
            field = torch.exp(-field ** 2 / (2 * sigma ** 2)) * risks[t]

            if self.args.pred_type == "multi_pred":
                field = field.view(B, M, -1).mean(dim=1)

            where_exist = torch.sum(field >= 0.01, axis=0) >= 2
            where_ind = torch.where(where_exist)[0]

            map_risk = torch.sum(field, axis=0)
            acc_risk[t] = map_risk[where_ind].mean()
            self.map_risk += map_risk
        acc_risk = torch.nan_to_num(acc_risk, nan=0)
        return acc_risk


    def forward_batch(self, tras):
        device = self.device
        thr = torch.tensor(0.01, device=device)
        denom = torch.sqrt(-2 * torch.log(thr))

        if self.args.pred_type != "multi_pred":
            risks_pair = self.risk_cal.forward(tras)  # [T, B, B]
            radius_batched = vmap(lambda x: self.cal.adj_dis(x, x))
            radius_pair = radius_batched(tras[:, :, :2])  # [T, B, B]

            radius = (radius_pair * (risks_pair > 0).float()).amax(dim=-1, keepdim=True)  # [T, B, 1]
            risks = risks_pair.mean(dim=-1, keepdim=True)  # [T, B, 1]
            pts = tras[:, :, :2]  # [T, B, 2]

            field_dist_batched = vmap(lambda x: self.cal.adj_dis(x, self.map))
            dists = field_dist_batched(pts)  # [T, B, M_map]

            sigma = radius / denom  # [T, B, 1]
            field = torch.exp(-(dists ** 2) / (2 * (sigma ** 2))) * risks  # [T, B, M_map]

            where_exist = (field >= thr).sum(dim=1) >= 2  # [T, M_map]
            map_risk = field.sum(dim=1)  # [T, M_map]

            acc_risk = (map_risk * where_exist.float()).sum(dim=1) / where_exist.sum(dim=1)  # [T]
            acc_risk = torch.nan_to_num(acc_risk, nan=0.0)

            self.map_risk += map_risk.sum(dim=0)
            return acc_risk

        else:
            # 多模态
            T, B, M, F = tras.shape

            risks_modes = [self.risk_cal.forward(tras[:, :, m]) for m in range(self.args.K)]  # 每项 [T,B,B]
            risks_modes = torch.stack(risks_modes, dim=-1)  # [T, B, B, K]

            radius_batched = vmap(lambda x: self.cal.adj_dis(x, x))
            radius_modes = [radius_batched(tras[:, :, m, :2]) for m in range(self.args.K)]  # 每项 [T,B,B]
            radius_modes = torch.stack(radius_modes, dim=-1)  # [T, B, B, K]

            radius_modes = (radius_modes * (risks_modes > 0).float()).amax(dim=2)  # [T, B, K]
            risks_modes = risks_modes.mean(dim=2)  # [T, B, K]

            pts = tras[..., :2].view(T, -1, 2)  # [T, B*K, 2]
            radius = radius_modes.view(T, -1, 1)  # [T, B*K, 1]
            risks = risks_modes.view(T, -1, 1)  # [T, B*K, 1]

            field_dist_batched = vmap(lambda x: self.cal.adj_dis(x, self.map))
            dists = field_dist_batched(pts)  # [T, B*K, M_map]

            sigma = radius / denom  # [T, B*K, 1]
            field = torch.exp(-(dists ** 2) / (2 * (sigma ** 2))) * risks  # [T, B*K, M_map]

            field = field.view(T, B, M, -1).mean(dim=2)  # [T, B, M_map]

            where_exist = (field >= thr).sum(dim=1) >= 2  # [T, M_map]
            map_risk = field.sum(dim=1)  # [T, M_map]

            acc_risk = (map_risk * where_exist.float()).sum(dim=1) / where_exist.sum(dim=1)  # [T]
            acc_risk = torch.nan_to_num(acc_risk, nan=0.0)

            self.map_risk += map_risk.sum(dim=0)

            return acc_risk


class Mapping_ST(Mapping_base):
    def __init__(self, args):
        super().__init__(args)

    def acc_prob(self, field, where_ind):
        sce = field[:, where_ind]
        accu = 1 - torch.prod(1 - sce, axis=0)
        for i in range(sce.shape[0]):
            temp = torch.clone(1 - sce)
            temp[i] = sce[i]
            accu -= torch.prod(temp, axis=0)

        field_out = torch.zeros_like(field[0])
        field_out[where_ind] = accu
        return field_out


    def forward_time(self, tras):

        if not self.args.pred_type == "multi_pred":
            T, B, F = tras.shape
            radius = (tras[..., -1] * self.args.sd_scale[0]).unsqueeze(-1)
            tras = tras[..., :2]
        else:
            T, B, M , F = tras.shape
            radius = (tras[..., -1] * self.args.sd_scale[0]).view(T, -1, 1)
            tras = tras[..., :2].view(T, -1, 2)

        acc_risk = torch.zeros(T, device=tras.device)
        for t in range(T):
            field = self.cal.adj_dis(tras[t], self.map)
            sigma = (radius[t] / torch.sqrt(-2 * torch.log(torch.tensor(0.01, device=tras.device))))
            field = torch.exp(-field ** 2 / (2 * sigma ** 2))

            if self.args.pred_type == "multi_pred":
                field = field.view(B, M, -1).mean(dim=1)

            where_exist = torch.sum(field >= 0.01, axis=0) >= 2
            where_ind = torch.where(where_exist)[0]

            map_risk = self.acc_prob(field, where_ind)
            acc_risk[t] = map_risk[where_ind].mean()
            self.map_risk += map_risk
        acc_risk = torch.nan_to_num(acc_risk, nan=0.0)
        return acc_risk


    def forward_batch(self, tras):

        device = tras.device
        thr = 0.01
        sconst = torch.sqrt(-2.0 * torch.log(torch.tensor(thr, device=device, dtype=tras.dtype)))
        scale0 = self.args.sd_scale[0]

        if self.args.pred_type != "multi_pred":
            T, B, F = tras.shape
            radius = (tras[..., -1] * scale0).unsqueeze(-1)  # (T,B,1)
            pos = tras[..., :2].reshape(T * B, 2)  # (T*B,2)
            dist = self.cal.adj_dis(pos, self.map)  # (T*B,P)
            P = dist.shape[-1]
            dist = dist.view(T, B, P)  # (T,B,P)
            sigma = radius / sconst  # (T,B,1)
            field = torch.exp(-(dist ** 2) / (2.0 * sigma ** 2))  # (T,B,P)
        else:
            T, B, M, F = tras.shape
            radius = (tras[..., -1] * scale0).unsqueeze(-1)  # (T,B,M,1)
            pos = tras[..., :2].reshape(T * B * M, 2)  # (T*B*M,2)
            dist = self.cal.adj_dis(pos, self.map)  # (T*B*M,P)
            P = dist.shape[-1]
            dist = dist.view(T, B, M, P)  # (T,B,M,P)
            sigma = radius / sconst  # (T,B,M,1)
            field = torch.exp(-(dist ** 2) / (2.0 * sigma ** 2))  # (T,B,M,P)
            field = field.mean(dim=2)  # (T,B,P) 先按M平均

        mask = (field >= thr).sum(dim=1) >= 2  # (T,P)

        p = field  # (T,B,P)
        R = 1 - p  # (T,B,P)
        P0 = torch.prod(R, dim=1)  # (T,P)

        prefix = torch.cumprod(R, dim=1)  # (T,B,P)
        suffix = torch.cumprod(R.flip(dims=[1]), dim=1).flip(dims=[1])
        ones = torch.ones((prefix.size(0), 1, prefix.size(2)), device=device, dtype=field.dtype)
        prefix_excl = torch.cat([ones, prefix[:, :-1, :]], dim=1)  # (T,B,P)
        suffix_excl = torch.cat([suffix[:, 1:, :], ones], dim=1)  # (T,B,P)
        prod_excl = prefix_excl * suffix_excl  # (T,B,P)

        P1 = torch.sum(p * prod_excl, dim=1)  # (T,P)
        map_risk = 1 - P0 - P1  # (T,P)
        map_risk = map_risk * mask.to(map_risk.dtype)

        counts = mask.sum(dim=1).to(map_risk.dtype)  # (T,)
        sums = map_risk.sum(dim=1)  # (T,)
        acc_risk = torch.where(counts > 0, sums / counts, torch.zeros_like(sums))
        acc_risk = torch.nan_to_num(acc_risk, nan=0.0)

        self.map_risk = self.map_risk + map_risk.sum(dim=0)  # (P,)

        return acc_risk








class Mapping_SD(Mapping_CRI):
    def __init__(self, args):
        super().__init__(args)
        self.risk_cal = SD_Cal(args)


class Mapping_SD2(Mapping_CRI):
    def __init__(self, args):
        super().__init__(args)
        self.risk_cal = SD_Cal2(args)

class Mapping_CRI2(Mapping_CRI):
    def __init__(self, args):
        super().__init__(args)
        self.risk_cal = CRI_Cal2(args)