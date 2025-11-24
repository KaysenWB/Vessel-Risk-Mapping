import numpy as np
import matplotlib.pyplot as plt
from utils import Calculating
import pickle
from tqdm import tqdm
from risk_calculating import CRI_Cal, CRI_Cal2, SD_Cal, SD_Cal2
from risk_calculating_map import *
from utils import Class_Indicator
import torch
import pandas as pd
from scipy.interpolate import interp1d

class Risk_Preception:
    def __init__(self, args):
        self.args = args
        self.device = args.device
        self.data_path = f"{args.pred_dir}/{args.model}_Pred.npy"
        self.save_path = f"{args.result_dir}/{args.flag}_{args.method}.npy"
        self.event_path = f"{args.event_dir}/{args.flag}.npy"
        self.alarm_adv = args.alarm_adv
        self.flag = args.flag
        self.K = args.K
        self.read_data()
        self.build_method()
        self.cal = Calculating()

    def read_data(self):
        self.dataset = torch.tensor(np.load(self.data_path), device=self.device)
        self.events = torch.tensor(np.load(self.event_path), device=self.device)
        inver = self.events.shape[0] // self.args.cal_samples
        self.events = self.events[::inver]
        self.events = torch.cat((self.events.chunk(2, dim=-1)), dim=0).long()
        self.events[:, -1] = self.events[:, -1] - self.alarm_adv
        self.events_num = self.events.shape[0]

        #self.holds = torch.cat((torch.linspace(0, 0.075, 4), torch.linspace(0.1, 1, 10)), dim=0)
        self.holds = torch.linspace(0, 1, 101).to(self.device)
        self.map_holds = torch.cat((torch.linspace(0, 0.2, 101).to(self.device), self.holds[20:]), dim=-1)

        if self.args.method in ["cri", "sd", "cri2", "sd2"]:
            self.holds_len = len(self.holds)
        else:
            self.holds_len = len(self.map_holds)

    def build_method(self):
        method_dict = {"cri": CRI_Cal,
                       "cri2": CRI_Cal2,
                       "sd": SD_Cal,
                       "sd2": SD_Cal2,
                       "map_cri2": Mapping_CRI2,
                       "map_sd2": Mapping_SD2,
                       "map_cri": Mapping_CRI,
                       "map_sd": Mapping_SD,
                       "map_st": Mapping_ST,

        }
        self.method = method_dict[self.args.method](self.args)

    def stimulating(self, event, event_steps):
        adv_steps = event.shape[2] - 1
        predicted = torch.zeros_like(event)[:event_steps, :, :adv_steps]
        for t in range(event_steps):
            for a in range(adv_steps):
                predicted[t, :, a] = event[t + a, :, a, :]
        return predicted.permute(0, 2, 1, 3)

    def insert_func(self, ser, rep, kind):

        x = np.linspace(0, 1, len(ser))
        x_in = np.linspace(0, 1, rep)
        func = interp1d(x, ser, kind=kind)

        return func(x_in)

    def mtras_samples(self, tras):
        t_, b_, f_ = tras.shape
        tras = tras.unsqueeze(2).repeat(1, 1, self.K, 1)

        position_noise = torch.normal(0, 0.001, (t_, b_, self.K - 1, 2)).to(self.device)
        speed_noise = torch.normal(0, 0.05, (t_, b_, self.K - 1, 1)).to(self.device) * 10
        direction_noise = torch.normal(0, 0.20, (t_, b_, self.K - 1, 1)).to(self.device) * 100
        noise = torch.cat([position_noise, speed_noise, direction_noise], dim=-1)
        cumulative_noise = torch.cumsum(noise, dim=0) * 0.1

        tras[:, :, :-1, :-1] += cumulative_noise
        return tras

    def show_mtras(self, mtras):
        mtras = mtras.numpy()
        t_, b_, k, _ = mtras.shape
        fig, ax = plt.subplots(2, 2, figsize= (16, 9))
        for i in range(b_):
            ax[0, 0].plot(mtras[:, i, :, 0], mtras[:, i, :, 1], lw = 0.5)
        ax[0, 0].imshow(plt.imread("/Users/yangkaisen/MyProject/Data/map/hongkong.jpg"),
                  extent=[114.099003, 114.187537, 22.265695, 22.322062])
        xx = np.linspace(0, 1, t_)
        ob_ship = 0
        ax[1, 0].plot(xx, mtras[:, ob_ship, 0, 2], lw=0.5, c= "k")
        ax[1, 1].plot(xx, mtras[:, ob_ship, 0, 3], lw=0.5, c= "k")
        ax[1, 0].plot(xx, mtras[:, ob_ship, 1:, 2], lw = 0.5)
        ax[1, 1].plot(xx, mtras[:, ob_ship, 1:, 3], lw=0.5)
        plt.show()
        return

    def risk_cal(self, tras):
        # risks cal
        if self.args.pred_type == "multi_pred":
            mtras = self.mtras_samples(tras)
            if self.args.method in ["cri", "sd", "cri2", "sd2"]:
                risks = [self.method.forward(mtras[:, :, m]) for m in range(self.K)]
                # assume each mode has the same probability
                risks = torch.stack(risks, dim=-1).amax(dim=-1)
            else:
                risks = self.method.forward(mtras)
        else:
            risks = self.method.forward(tras)

        # warning
        if self.args.method in ["cri", "sd", "cri2", "sd2"]:
            risks = risks.amax(dim=(1, 2))
            alarm = risks.unsqueeze(1) > self.holds.unsqueeze(0)
        else:
            alarm = risks.unsqueeze(1) > self.map_holds.unsqueeze(0)
        # are there four(alarm_time) consecutive warnings
        warn = torch.any(alarm.unfold(0, self.args.alarm_time, 1).all(dim=-1), dim=0)
        return warn


    
    def forward(self):
        event_warns = torch.zeros((self.events_num, self.holds_len), dtype=bool, device=self.device)
        for e, event in enumerate(tqdm(self.events)):
            event = event[event != 0]
            event_steps = event[-1]
            event_tra = self.dataset[:, event[:-1]]
            #real = event_tra[:,:,0,:]
            #see = self.risk_cal(real)
            preds = self.stimulating(event_tra, event_steps)
            step_warns = torch.zeros((event_steps, self.holds_len), dtype=bool, device=self.device)
            for p in range(event_steps):
                step_warns[p] = self.risk_cal(preds[p])
            # are there four (warn_time) consecutive warnings
            event_warns[e] = torch.any(step_warns.unfold(0, self.args.warn_time, 1).all(dim=-1), dim=0)

        event_warns = torch.stack((event_warns.chunk(2)), dim=0).cpu().numpy()
        np.save(self.save_path, event_warns)
        print(f"Preds: {self.args.model}, Saved Flag: {self.flag}, Method: {self.args.method}")


    def report(self):
        metrics = np.load(self.save_path)#[:, :self.args.report_samples]
        positives, negative = metrics

        TP = np.sum(positives, axis=0)
        FN = self.events_num // 2 - TP

        FP = np.sum(negative, axis=0)
        TN = self.events_num//2 - FP

        reporter = Class_Indicator(TP, FN, FP, TN)
        reports = reporter.forward()
        self.key = np.array(list(reports.keys()))
        whole_re = np.array(list(reports.values()))

        condi = (whole_re[3] <= 0.1).astype("float32")
        ind = (whole_re[0] * whole_re[-2])
        best_re = whole_re[:, np.argmax(condi * ind)]
        return best_re, whole_re

    def save_reports(self, save_reports):
        report, whole, nalist, molist = save_reports
        table = pd.DataFrame(report, columns=self.key)
        table.insert(0, "method", nalist)
        table.insert(0, "pred_model", molist)

        n_keys = len(self.key)
        nalist = np.array(nalist, dtype=object).repeat(n_keys)[:, np.newaxis]
        molist = np.array(molist, dtype=object).repeat(n_keys)[:, np.newaxis]

        whole = [np.concatenate((self.key[:, np.newaxis], w), axis=-1).astype(object) for w in whole]
        whole = np.concatenate(whole, axis=0)
        whole = np.concatenate((molist, nalist, whole), axis=-1)
        whole_df = pd.DataFrame(whole)

        if self.args.pred_type == "multi_pred":
            table.to_csv("./result/reports_table_m.csv", index=False)
            whole_df.to_csv("./result/whole_reports_m.csv", index=False)
        else:
            table.to_csv("./result/reports_table.csv", index=False)
            whole_df.to_csv("./result/whole_reports.csv", index=False)
        return








