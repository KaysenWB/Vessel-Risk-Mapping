import pandas as pd
import torch
import numpy as np
import pickle
import numpy as np
import matplotlib.pyplot as plt





class Class_Indicator:
    def __init__(self, TP, FN, FP, TN):
        self.TP = TP
        self.FN = FN
        self.FP = FP
        self.TN = TN
        self.all = self.TP + self.FN + self.FP + self.TN
        #               预测为正例    预测为负例
        # 实际为正例         TP          FN
        # 实际为负例         FP          TN

    def accuracy(self):
        # 报告正确（有无风险）的比例，越高越好
        return (self.TP + self.TN) / self.all

    def precision(self):
        # 报告有风险时这个报告的可信度，越高越好
        return self.TP / (self.TP + self.FP + 1e-6)

    def recall(self):
        # 越高越好，越高漏报越少
        return self.TP / (self.TP + self.FN + 1e-6)

    def fnr(self):
        # false negative rate
        # 漏报率
        return 1 - self.recall()

    def fpr(self):
        # false positive rate
        # 误报率
        # 所有无风险情况下报告了风险的比例，越低越好
        return self.FP / (self.FP + self.TN + 1e-6)

    def f1score(self):
        #系统在避免漏报和减少误报之间的平衡能力， 可以用来确定各类方法的参数，选F1表现最好的参数
        prec = self.precision()
        rec = self.recall()
        return 2 * (prec * rec) / (prec + rec + 1e-6)

    def kappa(self):
        """计算Cohen's Kappa系数"""
        Po = self.accuracy()

        # 计算边际概率
        p_actual_yes = (self.TP + self.FN) / self.all
        p_actual_no = (self.TN + self.FP) / self.all
        p_pred_yes = (self.TP + self.FP) / self.all
        p_pred_no = (self.TN + self.FN) / self.all

        # 期望一致率
        Pe = p_actual_yes * p_pred_yes + p_actual_no * p_pred_no

        return (Po - Pe) / (1 - Pe + 1e-6)

    def forward(self):
        report_dict_ = {"accuracy": self.accuracy(),
                       "precision": self.precision(),
                       "recall": self.recall(),
                       "fnr": self.fnr(),
                       "fpr": self.fpr(),
                       "f1score": self.f1score(),
                       "kappa": self.kappa(),}

        report_dict = {}
        for key, value in report_dict_.items():
            report_dict[key] = np.round(value, 3)
        return report_dict




class Calculating:

    def heading_torch(self, coor):
        if coor.dim() == 2:
            coor = coor.permute(1, 0)
        else:
            coor = coor.permute(2, 1, 0)
        lon1, lat1, lon2, lat2 = torch.deg2rad(coor)
        delta_lon = lon2 - lon1
        y = torch.sin(delta_lon) * torch.cos(lat2)
        x = torch.cos(lat1) * torch.sin(lat2) - torch.sin(lat1) * torch.cos(lat2) * torch.cos(delta_lon)
        headings_rad = torch.atan2(y, x)
        headings_deg = torch.rad2deg(headings_rad)
        headings_deg = (headings_deg + 360)
        headings_deg = (headings_deg + 360) % 360

        return headings_deg

    def heading_np(self, coor):

        lon1, lat1, lon2, lat2 = np.deg2rad(coor.T)
        delta_lon = lon2 - lon1

        y = np.sin(delta_lon) * np.cos(lat2)
        x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(delta_lon)

        headings_rad = np.arctan2(y, x)
        headings_deg = np.degrees(headings_rad)
        headings_deg = (headings_deg + 360) % 360

        return headings_deg

    def distance_torch(self, coor):
        if coor.dim() == 2:
            coor = coor.permute(1, 0)
        else:
            coor = coor.permute(2, 1, 0)
        lon1, lat1, lon2, lat2 = torch.deg2rad(coor)
        dlon = lon2 - lon1
        dlat = lat2 - lat1

        a = torch.sin(dlat / 2) ** 2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2) ** 2
        c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))
        return  c * 6371.0 # Unit: Km

    def distance_np(self, coor):

        lon1, lat1, lon2, lat2 = np.deg2rad(coor.T)
        dlon = lon2 - lon1
        dlat = lat2 - lat1

        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        return  c * 6371.0 # Unit: Km

    def heading(self, coor):
        # from fist(start) point to second point(end)
        if isinstance(coor, torch.Tensor):
            return self.heading_torch(coor)
        else:
            return self.heading_np(coor)

    def distance(self, coor):
        # Unit: m
        if isinstance(coor, torch.Tensor):
            dis = self.distance_torch(coor) * 1000
            return dis
        else:
            dis = self.distance_np(coor) * 1000
            return dis

    def adj_dis(self, p1, p2):
        if isinstance(p1, torch.Tensor):
            out = self.adj_dis_torch(p1, p2) * 1000
        else:
            out = self.adj_dis_np(p1, p2) * 1000
        return out

    def time_adj_dis(self, data):
        if isinstance(data, torch.Tensor):
            out = self.time_adj_dis_tensor(data)  * 1000
        else:
            out = self.time_adj_dis_np(data)  * 1000
        return out



    def adj_dis_np(self, p1, p2):
        times1, times2 = len(p2), len(p1)
        p1 = np.repeat(p1[np.newaxis, :, :], times1, axis=0)
        p2 = np.repeat(p2[:, np.newaxis, :], times2, axis=1)
        adj_dis = self.distance_np(np.concatenate((p1, p2), axis=-1))
        return adj_dis

    def adj_dis_torch(self, p1, p2):
        times1, times2 = len(p2), len(p1)
        p1 = p1.unsqueeze(0).repeat(times1, 1, 1)
        p2 = p2.unsqueeze(1).repeat(1, times2, 1)
        adj_dis = self.distance_torch(torch.cat((p1, p2), axis=-1))
        return adj_dis



    def time_adj_dis_np(self, data):
        # data [T, B, F], perfer numpy
        T, B, F = data.shape
        adj = np.zeros((T, B, B))
        for t in range(T):
            adj[t] = self.adj_dis_np(data[t], data[t])
        return adj

    def time_adj_dis_tensor(self, data):
        # data [T, B, F], perfer tensor
        T, B, F = data.shape
        adj = torch.zeros((T, B, B), device= data.device)
        for t in range(T):
            adj[t] =self.adj_dis_torch(data[t], data[t])
            #step = data[t]
            #adj[t] = self.distance_torch(torch.cat((step.repeat(B, 1, 1), step.unsqueeze(1).repeat(1, B, 1)), dim=-1))
        return adj


    def coor_project_paras(self, ext):

        ext = np.array(ext)
        diff_lon = ext[1] - ext[0]
        diff_lat = ext[3] - ext[2]

        nodes_left_low = ext[[0, 2]]
        nodes_right_low = ext[[1, 2]]
        nodes_left_high = ext[[0, 3]]
        nodes_right_high = ext[[1, 3]]

        dis_lon = self.distance(np.concatenate((nodes_left_low, nodes_right_low)))
        dis_lon_ = self.distance(np.concatenate((nodes_left_high, nodes_right_high)))
        dis_lon = (dis_lon + dis_lon_) * 0.5 * 0.01

        dis_lat = self.distance(np.concatenate((nodes_left_low, nodes_left_high)))
        dis_lat_ = self.distance(np.concatenate((nodes_right_low, nodes_right_high)))
        dis_lat = (dis_lat + dis_lat_) * 0.5 * 0.01

        para_lon = diff_lon / dis_lon
        para_lat = diff_lat / dis_lat

        return para_lon, para_lat



class Water_indicator:
    def __init__(self):
        self.data = None

    def count(self):
        """
            A day: ships: {1114}, traces: {6902},  length: {44.85}
            A week: ships: {2401}, traces: {25503},  length: {44.79}
            A month: ships: {4795}, traces: {31939},  length: {43.19}
        """
        title = self.data['title']
        da = self.data['data_np']
        ships = len(set(da[:, 6]))
        traces = len(set(da[:, 1]))
        length = da[:, 8].mean()

        return f"ships: {ships}, traces: {traces},  length: {length}"


    def read_date(self, root):
        suffix = root.split('.')[-1]

        if suffix == 'csv':
            self.data = pd.read_csv(root)
        elif suffix == 'npy':
            self.data = np.load(root, allow_pickle=True)

        elif suffix == 'pt':
            self.data = torch.load(root)

        elif suffix == 'pkl' or suffix == 'cpkl' :
            with open(root, 'rb') as f:
                self.data = pickle.load(f)
        return self.data

    def re_save(self, root, suffix = '.pt'):
        r_ = root.split('.')[0]
        re_r = r_ + suffix

        if not self.data:
            self.read_date(root)

        if suffix == '.csv':
            if not isinstance(self.data, pd.DataFrame):
                self.data = pd.DataFrame(self.data)
                ### change for spcial tasks
            with open(re_r, 'rb') as f:
                self.data.to_csv(f, index_label=False)

        elif suffix == '.npy':
            if not isinstance(self.data, np.ndarray):
                self.data = np.array(self.data)
                ### change for spcial tasks
            np.save(re_r, self.data)

        elif suffix == '.pt':
            ### change for spcial tasks
            torch.save(self.data, re_r)
        elif suffix == '.pkl' or suffix == '.cpkl':
            ### change for spcial tasks
            dict_data = {'title': self.data.columns, 'data_np': self.data.values}
            with open(re_r, 'wb') as f:
                pickle.dump(dict_data, f)

class Risk_Mapping:
    def __init__(self, args):
        self.args = args
        self.device = args.device
        self.cal = Calculating()
        self.initial()

    def initial(self):
        self.dataset = np.load(f"{self.args.pred_dir}/{self.args.model}_Pred.npy")
        self.dataset = self.dataset[48:]
        lon_min, lon_max, lat_min, lat_max = 114.099003, 114.187537, 22.265695, 22.322062
        lon_grid = np.linspace(lon_min, lon_max, 250)
        lat_grid = np.linspace(lat_min, lat_max, 200)
        Lon, Lat = np.meshgrid(lon_grid, lat_grid)
        self.risk_map = np.stack((Lon, Lat), axis=-1).reshape(-1, 2)

    def forward(self, flag="coll", use_new=False):
        self.risks = np.zeros_like(self.risk_map)[:, 0]
        r_flag = "new" if use_new else "trad"

        with open(f'mixed_{flag}.pkl', 'rb') as f:
            self.events = pickle.load(f)

        keep = [1, 2, 4, 6, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 35, 36, 38, 54]
        event_tra = self.dataset[:, keep]
        real = event_tra[:32, :, 0]
        pred = event_tra[:32, :, 12]
        #pred_ = torch.stack([event_tra[i, :, i] for i in range(32)])

        for i, pred_p in enumerate(pred):
            dis = self.cal.adj_dis(pred_p, self.risk_map[:, :2])

            if use_new:
                indices = np.where(np.sum(dis <= 100, axis=0) >= 2)[0]
                pred_p_new = self.risk_map[indices]
                dis = self.cal.adj_dis(pred_p_new, self.risk_map)
                dis = 10 / (dis + 30)
            else:
                wig = self.cal.adj_dis(pred_p, pred_p)
                wig = np.sum(1 / (wig + 1), axis=0) - 1
                wig = np.where(wig < np.percentile(wig, 30), 0, 1)[:, np.newaxis]
                dis = (10 / (dis + 30)) * wig

            dis = np.sum(dis, axis=0) - 0.1
            risks = np.clip(dis, 0, 1)
            self.risks += risks

            if i % 4 == 0:
                self.show_pics(risks, real, pred, i, r_flag)

        self.risks = (self.risks - self.risks.min()) / (self.risks.max() - self.risks.min())
        self.show_pics(self.risks, real, pred, i, r_flag)

    def show_pics(self, risks, real, pred, i, r_flag):

        plt.figure(figsize=(10, 7))
        plt.scatter(self.risk_map[:, 0], self.risk_map[:, 1], c="red", alpha=risks, s=5, zorder=0)
        plt.plot(real[:, :, 0], real[:, :, 1], color="blue", lw=1, zorder=1)
        plt.plot(pred[:, :, 0], pred[:, :, 1], color="green", lw=1, zorder=2)

        imp = plt.imread("/Users/yangkaisen/MyProject/Data/map/hongkong2.jpg")
        plt.imshow(imp, extent=[114.099003, 114.187537, 22.265695, 22.322062], zorder=3)
        plt.savefig(f"pics_risk/{r_flag}_{i}.jpg", dpi=300, bbox_inches='tight')
        plt.show()
        print("saved")



if __name__ == "__main__":
    cal = Calculating()
    points = np.array((114.18503, 22.29018, 114.19169, 22.29558))
    heading = cal.heading(coor=points)
    print(';')