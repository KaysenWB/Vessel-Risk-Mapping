import numpy as np
import matplotlib.pyplot as plt
from utils import Calculating
import torch
import torch.nn as nn

class Risk_Base:
    def __init__(self, args):
        self.cal = Calculating()
        self.inver = args.inver
        self.conv_sc_og = 7
        self.conv_dt_cpa = 7
        self.ave_ship = 50
        self.project_para = torch.tensor(args.project_para).to(args.device)
        self.d_range = args.d_range
        self.t_range = args.t_range

    def smooth(self, seq, k=7):
        # tensor [T, B] 或 [T]
        kernel = torch.ones(1, 1, k, device=seq.device) / k

        if seq.ndim > 1:
            for i in range(seq.shape[1]):
                seq[:, i] = torch.nn.functional.conv1d(
                    seq[:, i].view(1, 1, -1),
                    kernel,
                    padding=k // 2
                ).squeeze()
        else:
            seq = torch.nn.functional.conv1d(
                seq.view(1, 1, -1),
                kernel,
                padding=k // 2
            ).squeeze()
        return seq

    def complete(self, tras):
        # tras[T, B, F]
        if tras.shape[-1] == 2:
            scr = tras[:-1]
            tar = tras[1:]
            coor = torch.cat((scr, tar), -1)

            dis = self.smooth(self.cal.distance(coor).permute(1, 0), k=self.conv_sc_og).unsqueeze(-1)
            heading = self.smooth(self.cal.heading(coor).permute(1, 0), k=self.conv_sc_og).unsqueeze(-1)  # 从正北开始算的角
            # dis = self.cal.distance(coor).mT.unsqueeze(-1)
            # heading = self.cal.heading(coor).mT.unsqueeze(-1)  # 从正北开始算的角

            speed = dis / self.inver  # m/s
            size = torch.ones_like(heading) * self.ave_ship
            tras_sh = torch.cat((scr, speed, heading, size), axis=-1)
            # self.vaild_speed_head(tras_sh)
        else:
            tras[:, :, 2] =tras[:,:, 2] *  0.5144
            tras_sh = tras

        return tras_sh

    def risk_cal(self, tras):
        return

    def pair_risk(self, tras):
        T, B, _ = tras.shape
        ros, cls = torch.tril_indices(B, B, offset=-1)
        risks = torch.zeros((T, B, B), device=tras.device)

        for ros_idx, cls_idx in zip(ros, cls):
            r = self.risk_cal(tras[:, [ros_idx, cls_idx]])
            risks[:, ros_idx, cls_idx] = r
            risks[:, cls_idx, ros_idx] = r
        return risks


    def vaild_speed_head(self, tras_sh):
        tras_sh = tras_sh.numpy()
        obs_point = 10
        ax = self.vaild(tras_sh, obs_point)
        xx = range(len(tras_sh))
        for i, idx in enumerate([3, 2]):
            for j, color in enumerate(['k', 'b']):
                ax[1, i].plot(xx, tras_sh[:, j, idx], c=color)
                ax[1, i].scatter(xx[obs_point], tras_sh[obs_point, j, idx], c="red")
        #plt.savefig("./vaild_test.jpg", dpi = 500)
        plt.show()
        return

    def vaild(self, tras, obs_point = 0):

        fig, ax = plt.subplots(3, 2, figsize=(14, 12))
        for i in [0, 1]:
            ax[0, i].scatter(tras[:, 0, 0], tras[:, 0, 1], c="k", s = 3)
            ax[0, i].scatter(tras[:, 1, 0], tras[:, 1, 1], c="b", s = 3)
            ax[0, i].scatter(tras[obs_point, 0, 0], tras[obs_point, 0, 1], c="r", s = 3)
            ax[0, i].scatter(tras[obs_point, 1, 0], tras[obs_point, 1, 1], c="r", s = 3)
        ax[0, 0].imshow(np.ones((10, 10, 4)), extent=[114.099003, 114.187537, 22.265695, 22.322062])
        return ax



class DTcpa_Cal(Risk_Base):
    def __init__(self, args):
        super().__init__(args)

    def cri_formula(self, d, t):
        return d, t

    def risk_cal(self, tras, re_cri = True):
        node1, node2 = tras[:, 0], tras[:, 1]
        lon1, lat1, sog1, cog1 = node1[:, 0], node1[:, 1], node1[:, 2], node1[:, 3]
        lon2, lat2, sog2, cog2 = node2[:, 0], node2[:, 1], node2[:, 2], node2[:, 3]

        v1_x, v1_y = sog1 * torch.sin(cog1 / 180 * torch.pi), sog1 * torch.cos(cog1 / 180 * torch.pi)
        v2_x, v2_y = sog2 * torch.sin(cog2 / 180 * torch.pi), sog2 * torch.cos(cog2 / 180 * torch.pi)

        vr_x, vr_y = v2_x - v1_x, v2_y - v1_y
        vr = torch.sqrt(vr_x ** 2 + vr_y ** 2)

        wr = torch.rad2deg(torch.atan2(vr_x, vr_y))
        wr[wr < 0] += 360

        coor = torch.cat((node1[:, :2], node2[:, :2]), axis=-1)
        Dr = self.cal.distance(coor)  # m
        aT = self.cal.heading(coor)
        angle_rad = (wr - aT - 180) / 180 * torch.pi

        dcpa = Dr * torch.sin(angle_rad) # m
        tcpa = (Dr * torch.cos(angle_rad) / (vr + 1e-6)) / 60  # s

        dcpa = torch.abs(dcpa)
        #dcpa = self.smooth(torch.abs(dcpa), k=self.conv_dt_cpa)
        #tcpa = self.smooth(tcpa, k=self.conv_dt_cpa)
        #self.vaild_dtcpa_pics(self.cri_formula(dcpa, tcpa, Dr), tcpa, tras)
        if re_cri:
            return self.cri_formula(dcpa, tcpa, Dr)
        else:
            return dcpa, tcpa


    def vaild_dtcpa_pics(self, dcpa, tcpa, tras):
        dcpa, tcpa = dcpa.numpy(), tcpa.numpy()
        obs_point = 5
        ax = self.vaild(tras, obs_point)

        xx = range(len(dcpa))
        ax[1, 0].plot(xx, dcpa);
        ax[2, 0].plot(xx, tcpa);
        ax[1, 0].scatter(xx[obs_point], dcpa[obs_point], c="r")
        ax[2, 0].scatter(xx[obs_point], tcpa[obs_point], c="r")

        cog1, sog1 = tras[:, 0, 3], tras[:, 0, 2]
        cog2, sog2 = tras[:, 1, 3], tras[:, 1, 2]
        for y1, y2, a in zip([cog1, sog1], [cog2, sog2], [ax[1, 1], ax[2, 1]]):
            a.plot(xx, y1, c="k");
            a.plot(xx, y2, c="b")
            a.scatter(xx[obs_point], y1[obs_point], c="r");
            a.scatter(xx[obs_point], y2[obs_point], c="r")
        plt.show()
        return


    def forward(self, tras):
        tras = self.complete(tras)
        cri = self.pair_risk(tras)
        return cri


class CRI_Cal(DTcpa_Cal):
    def __init__(self, args):
        super().__init__(args)

    def cri_formula(self, d, t, dr):
        t = torch.abs(t)

        ud = (1 - d / (self.d_range+ 1e-4)) * 0.5
        ut = (1 - t / (self.t_range + 1e-3)) * 0.5
        #cri = torch.clamp( ud, 0, 1) * torch.clamp(ut, 0, 1)
        cri = torch.clamp((ut + ud), 0, 1)
        return cri


class CRI_Cal2(DTcpa_Cal):
    def __init__(self, args):
        super().__init__(args)

    def cri_formula(self, d, t, dr):
        t = torch.abs(t)
        d1, d2 = 48, self.d_range # mean_ship_size
        t1, t2 = 1, self.t_range
        ud = (d / d2) ** 2
        ut = (t / t2) ** 2
        utr = (dr / d2) ** 2

        cri = 1 - torch.clamp(torch.sqrt(ud + ut + utr) , 0, 1)
        return cri



class SD_Cal(Risk_Base):
    def __init__(self, args):
        super().__init__(args)
        self.device = args.device
        self.grid_size = args.grid_size
        self.sd_scale = torch.tensor(args.sd_scale, device=self.device)

    def rotate_batch(self, x, y, heading):
        cos1 = torch.cos(heading)[:, None, None]
        sin1 = torch.sin(heading)[:, None, None]
        x_rot = x * cos1 + y * sin1
        y_rot = -x * sin1 + y * cos1
        return x_rot, y_rot

    def grids(self, tras, extent):
        x = torch.linspace(-extent, extent, self.grid_size, device=self.device)
        y = torch.linspace(-extent, extent, self.grid_size, device=self.device)
        x, y = torch.meshgrid(x, y, indexing='xy')

        n_times = tras.shape[0]
        x = x.expand(n_times, self.grid_size, self.grid_size)
        y = y.expand(n_times, self.grid_size, self.grid_size)
        return  x, y,


    def risk_cal(self, tras):

        # ship domain size
        size_s1, size_s2 = tras[0, :, 4]
        a1_val, b1_val = size_s1 * self.sd_scale
        a2_val, b2_val = size_s2 * self.sd_scale

        # distance of two domain
        diff = tras[:, 1, :2] - tras[:, 0, :2]
        diff = (diff / self.project_para) * 100

        # get map
        extent = a1_val + a2_val * 2
        x, y = self.grids(tras, extent)

        # domain1
        heading1 = torch.deg2rad(tras[:, 0, 3])
        x_rot1, y_rot1 = self.rotate_batch(x, y, heading1)
        a1_view = a1_val.view(1, 1, 1)
        b1_view = b1_val.view(1, 1, 1)
        domain1 = (x_rot1 ** 2 / a1_view ** 2 + y_rot1 ** 2 / b1_view ** 2) <= 1

        # domain2
        heading2 = torch.deg2rad(tras[:, 1, 3])
        x_shift = x - diff[..., 0].view(-1, 1, 1)
        y_shift = y - diff[..., 1].view(-1, 1, 1)
        x_rot2, y_rot2 = self.rotate_batch(x_shift, y_shift, heading2)
        a2_view = a2_val.view(1, 1, 1)
        b2_view = b2_val.view(1, 1, 1)
        domain2 = (x_rot2 ** 2 / a2_view ** 2 + y_rot2 ** 2 / b2_view ** 2) <= 1

        # overlap_ratio
        overlap_area = torch.sum(domain1 & domain2, dim=(1, 2))
        sum_domain1 = torch.sum(domain1, dim=(1, 2))
        sum_domain2 = torch.sum(domain2, dim=(1, 2))
        overlap_ratio = overlap_area / (torch.minimum(sum_domain1, sum_domain2) + 1)

        show = False
        if show:
            n_times = tras.shape[0]
            Domain1 = [domain1[i] for i in range(n_times)]
            Domain2 = [domain2[i] for i in range(n_times)]
            self.vaild_sdomain_pics(tras, overlap_ratio, Domain1, Domain2)
        return overlap_ratio

    def vaild_sdomain_pics(self, tras, risks, D1, D2):
        tras, risks = tras.numpy(), risks.numpy()
        obs_point = 46
        ax = self.vaild(tras, obs_point)
        d1, d2 = D1[obs_point], D2[obs_point]

        for i, (d_or, d_and) in enumerate([(d1 | d2, ~(d1 | d2)), (d1, ~d1)]):
            ax[1, i].scatter(*np.where(d_or), c='r', s = 0.4)
            ax[1, i].scatter(*np.where(d_and), c='grey', s = 0.1, alpha = 0.5)
            ax[1, i].set_aspect('equal')

        ax[2, 0].plot(risks, 'r')
        xx = range(len(tras))
        ax[2, 1].plot(xx, tras[:, 0, 3], 'k')
        ax[2, 1].plot(xx, tras[:, 1, 3], 'b')

        plt.show()
        return

    def forward(self, tras):
        # tras [Time, Batch, feats(lon, lat, sog, cog, size)]
        tras = self.complete(tras)
        risks = self.pair_risk(tras)
        return risks



class SD_Cal2(Risk_Base):
    def __init__(self, args):
        super().__init__(args)
        self.sd_scale = torch.tensor(args.sd_scale, device=args.device)


    def risk_cal(self, tras):

        size_s1, size_s2 = tras[0, :, 4]
        a1, b1 = size_s1 * self.sd_scale
        a2, b2 = size_s2 * self.sd_scale

        heading1 = torch.deg2rad(tras[:, 0, 3])
        heading2 = torch.deg2rad(tras[:, 1, 3])

        # relative position vector
        diff = tras[:, 1, :2] - tras[:, 0, :2]
        diff = (diff / self.project_para) * 100  # m

        # ship2_in_domain1
        cos1 = torch.cos(heading1)
        sin1 = torch.sin(heading1)
        x2_in_1 = diff[:, 0] * cos1 + diff[:, 1] * sin1
        y2_in_1 = -diff[:, 0] * sin1 + diff[:, 1] * cos1
        ship2_in_domain1 = (x2_in_1 ** 2 / a1 ** 2 + y2_in_1 ** 2 / b1 ** 2) <= 1

        # ship2_in_domain1
        cos2 = torch.cos(heading2)
        sin2 = torch.sin(heading2)
        x1_in_2 = (-diff[:, 0]) * cos2 + (-diff[:, 1]) * sin2
        y1_in_2 = -(-diff[:, 0]) * sin2 + (-diff[:, 1]) * cos2
        ship1_in_domain2 = (x1_in_2 ** 2 / a2 ** 2 + y1_in_2 ** 2 / b2 ** 2) <= 1

        overlap_ratio = (ship2_in_domain1 | ship1_in_domain2).float()

        return overlap_ratio

    def forward(self, tras):
        # tras [Time, Batch, feats(lon, lat, sog, cog, size)]
        tras = self.complete(tras)
        risks = self.pair_risk(tras)
        return risks


