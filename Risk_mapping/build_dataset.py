import argparse
import os.path
import numpy as np
import matplotlib.pyplot as plt
import torch
from utils import Calculating, Water_indicator
import pickle




def show_(flag, args):
    if flag == 0:
        pass
    if flag == 1:
        dataset = torch.tensor(np.load(f"{args.pred_dir}/{args.model}_Pred.npy"), device=args.device)
        Vis_ = Visual(args)
        Vis_.vis1(dataset)
    if flag == 2:
        sampler = SamplingCollMiss(args)
        sampler.show_coll_miss(flag='coll')
    if flag == 3:
        sampler = SamplingCollMiss(args)
        sampler.show_coll_miss(flag='miss')
    if flag == 4:
        sampler = SamplingCollMiss(args)
        sampler.show_mixed()




class Visual:
    def __init__(self, args):
        self.pla = 'pla'
        self.map_root = args.map_root

    def vis1(self, data, sh_=None):
        " a traject with 48/96 steps-advance prediction"
        if sh_ ==None:
            sh_ = [i for i in range(0, 150, 10)]
        timestep, _, adv_, _ = data.shape
        diff = 0.4 / timestep
        for sh in sh_:
            traj = data[:, sh, :, :]
            for i in range(1, adv_):
                #tra = traj[i, :, :]
                #plt.scatter(tra[:, 0], tra[:, 1], c='blue', s=1, alpha=0.05 + diff * i)
                tra = traj[:, -i, :]
                plt.plot(tra[:, 0], tra[:, 1], color='blue', lw=1, alpha=0.05 + diff * i)
                plt.plot(traj[:, 0, 0], traj[:, 0, 1], color='red', lw=1)

                #imp = plt.imread(self.map_root)
                #plt.imshow(imp, extent=[114.099003, 114.187537, 22.265695, 22.322062])
                plt.show()
                print(';')

    def vis2(self, tra1, tra2):
        "show pairs of collsions or near_miss"
        length = len(tra1)
        diff = 0.9 / length
        for t in range(length):
            plt.scatter(tra1[t, 0], tra1[t, 1], c='blue' , alpha=0.05+ t*diff, s=1)
            plt.scatter(tra2[t, 0], tra2[t, 1], c='red',alpha=0.05+ t*diff, s=1)
        imp = plt.imread(self.map_root)
        plt.imshow(imp, extent=[114.099003, 114.187537, 22.265695, 22.322062])
        plt.savefig(f'/Users/yangkaisen/MyProject/Risk_PartC/dataset/pics/_test.png', dpi=500, bbox_inches='tight')
        plt.show()
        print(';')

    def vis3(self, tras):
        color = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'black', 'plck', 'red', 'blue', 'green', 'yellow', 'purple', 'orange', 'black', 'plck']
        "show mixed scene"
        length = len(tras)
        diff = 0.9 / length

        for t in range(length):
            plt.scatter(tras[t, :, 0], tras[t, :, 1], c='blue', alpha=0.05+ t*diff, s=1)
        imp = plt.imread(self.map_root)
        plt.imshow(imp, extent=[114.099003, 114.187537, 22.265695, 22.322062])
        plt.show()
        print(';')

    def vis4(self, real, pred, id):
        "show remains preds in a timestep"
        length = len(pred)
        diff = 0.9 / length

        for t in range(length):
            plt.scatter(pred[t, :, 0], pred[t, :, 1], c='blue' , alpha=1 - t * diff, s=1)
        plt.plot(real[:, :, 0].numpy(), real[:, :, 1].numpy(), color='red', lw=1)
        imp = plt.imread(self.map_root)
        plt.imshow(imp, extent=[114.099003, 114.187537, 22.265695, 22.322062])
        plt.savefig(f'/Users/yangkaisen/MyProject/Risk_PartC/dataset/pics/_{id}', dpi=500, bbox_inches='tight')
        plt.show()
        print(';')
    def vis5(self, tras):
        "show infered speed and head"
        length, num, f = tras.shape
        tras = tras.cpu().numpy()
        diff = 0.9 / length
        for i in range(num):
            tra_ = tras[:, i]
            speed = tra_[:,  2]
            heading = tra_[:,  3]
            heading_rad = heading * (np.pi / 180)
            xx = tra_[:, 0] + np.cos(heading_rad) * 0.0007 #* ritio
            yy = tra_[:, 1] + np.sin(heading_rad) * 0.0001
            pointer = np.stack((xx, yy), axis=-1)
            for t in range(length-1):
                pp = np.stack((tra_[t, :2], pointer[t]), axis=0)
                plt.plot(pp[:, 0], pp[:, 1], color = 'purple', lw= 0.5, )
                plt.scatter(tra_[t, 0], tra_[t, 1], c='blue', alpha=0.1 + t * diff, s=1)
            #imp = plt.imread(self.map_root)
            #plt.imshow(imp, extent=[114.099003, 114.187537, 22.265695, 22.322062])
            plt.show()
            print(';')

        print(';')



class SamplingCollMiss:
    def __init__(self, args):
        self.args = args
        self.pred_dir = args.pred_dir
        self.model = args.model
        self.device = args.device
        self.sample_batch = 100
        self.samples = args.num_samples
        self.border = args.event_time_border
        self.coll_coef = args.coll_coef
        self.miss_coef = args.miss_coef
        self.ship_pairs_save = args.event_dir + "/pairs.npy"
        self.multi_ships_save = args.event_dir + "/multi.npy"
        self.cache()

    def cache(self):
        if not os.path.exists(self.ship_pairs_save):
            self.read_data()
            self.sampling_ship_pairs()
        if not os.path.exists(self.multi_ships_save):
            self.read_data()
            self.sampling_multi_ships()
        print("Cache finished")

    def read_data(self):
        self.dataset = torch.tensor(np.load(f"{self.pred_dir}/{self.model}_Pred.npy"), device=self.device)
        self.dataset = self.dataset[:, :, 0, :]
        self.traj_num = self.dataset.shape[1]
        self.cal = Calculating()


    def sampling_pairs(self, flag = "coll"):
        shots = torch.zeros((self.samples, 2), device=self.device)
        c = 0
        while c < self.samples:
            ind1 = torch.randint(0, self.traj_num, (self.sample_batch,))
            ind2 = torch.randint(0, self.traj_num, (self.sample_batch,))
            tra1 = self.dataset[:, ind1].squeeze()
            tra2 = self.dataset[:, ind2].squeeze()
            ship_size = (tra1[:, :, -1] + tra2[:, :, -1]).permute(1, 0)
            pair = torch.cat([tra1[:, :, :2], tra2[:, :, :2]], dim=-1)
            pair_dis = self.cal.distance(pair)

            if flag == 'coll':
                shot_ = pair_dis < ship_size * self.coll_coef
                extra_mask = torch.ones(shot_.shape[0], dtype=torch.bool, device=shot_.device)
            else:
                shot_ = (pair_dis < ship_size * self.miss_coef[1]) & (pair_dis > ship_size * self.miss_coef[0])
                extra_mask = ~torch.any(pair_dis < ship_size * self.miss_coef[0], dim=1)

            shot1 = torch.any(shot_[:, :self.border[0]], dim=1)
            shot2 = torch.any(shot_[:, self.border[0]: self.border[1]], dim=1)
            shot3 = torch.any(shot_[:, self.border[1]:], dim=1)

            ind_ = shot2 & ~shot1 & ~shot3 & extra_mask

            ind1_ = ind1[ind_]
            ind2_ = ind2[ind_]
            num_new = min(len(ind1_), self.samples - c)
            if num_new > 0:
                shots[c:c + num_new, 0] = ind1_[:num_new]
                shots[c:c + num_new, 1] = ind2_[:num_new]
                c += num_new
        return shots

    def identify_enent_time(self, events):
        src = self.dataset[:self.border[-1], events[:, 0], :2]
        tar = self.dataset[:self.border[-1], events[:, 1], :2]
        coor = torch.cat((src, tar), dim=-1)
        dis = self.cal.distance(coor)
        event_ind = torch.argmin(dis, dim=1).unsqueeze(1)
        events_new = torch.cat((events, event_ind), dim = -1)
        return events_new

    def sampling_ship_pairs(self, save = True):
        coll_p = self.sampling_pairs(flag="coll").long()
        miss_p = self.sampling_pairs(flag="miss").long()
        coll_p = self.identify_enent_time(coll_p)
        miss_p = self.identify_enent_time(miss_p)
        ship_pairs = torch.cat((coll_p, miss_p), dim=-1)
        if save:
            np.save(self.ship_pairs_save, ship_pairs.cpu().numpy())
            print('sampling ship pairs finished')
        else:
            return ship_pairs

    def add_one(self, event):
        news_ind = torch.randint(0, self.traj_num, (self.sample_batch,))
        news = self.dataset[:, news_ind].squeeze()

        shot_close = torch.zeros(self.sample_batch, dtype=torch.bool)
        shot_coll = torch.zeros(self.sample_batch, dtype=torch.bool)
        for e in event:
            ship = self.dataset[:, e]
            ship_size = (news[0, :, -1] + ship[0, -1]).unsqueeze(1)
            ship_tra = ship[:, :2].unsqueeze(1).repeat(1, self.sample_batch, 1)
            dis = torch.cat((ship_tra, news[:, :, :2]), dim=-1)
            dis = self.cal.distance(dis)

            shot_close_ = torch.any((dis < ship_size * self.miss_coef[1]), dim=-1)
            shot_coll_ = torch.any((dis < ship_size), dim=-1)

            shot_close = shot_close | shot_close_
            shot_coll = shot_coll | shot_coll_

        shot = shot_close & ~shot_coll
        ind = news_ind[shot]
        if not ind.numel() == 0:
            event = torch.cat((event, ind[:1]))
        return event

    def sampling_multi(self, base):
        samples_multi = torch.tensor(self.args.num_samples_multi, device=self.device)
        num_samples = torch.cumsum(samples_multi[:, 1], dim=0)
        max_ships = samples_multi[:, 0]

        group_ids = torch.searchsorted(num_samples, torch.arange(num_samples[-1], device=self.device))
        outputs = torch.zeros((self.samples, max_ships.max() + 1))
        outputs[:, -1] = base[:, -1]

        for eid, event in enumerate(base[:, :2]):
            group = group_ids[eid]
            while len(event) < max_ships[group]:
                event = self.add_one(event)
            outputs[eid, :len(event)] = event

        return outputs

    def sampling_multi_ships(self):
        base = self.sampling_ship_pairs(save=False)
        coll_m = self.sampling_multi(base[:, :3])
        miss_m = self.sampling_multi(base[:, 3:])

        multi_ships = torch.cat((coll_m, miss_m), dim=-1)
        np.save(self.multi_ships_save, multi_ships.cpu().numpy())
        print('sampling multi ships finished')
        return


    def vis_unit(self, ax, tras, shot, title):
        ax.plot(tras[:, :, 0], tras[:, :, 1], c="k")
        ax.scatter(tras[shot, :2, 0], tras[shot, :2, 1], c="r", s=15)
        ax.scatter(tras[0, :, 0], tras[0, :, 1], c="b", s=15)
        ax.imshow(plt.imread("/Users/yangkaisen/MyProject/Data/map/hongkong.jpg"),
                     extent=[114.099003, 114.187537, 22.265695, 22.322062])
        ax.set_title(title)
        return ax


    def vis_ship_pairs(self):
        self.read_data()
        events = np.load(self.ship_pairs_save)
        for i, event in enumerate(events):
            coll_tras = self.dataset[:, event[:2], :2].numpy()
            miss_tras = self.dataset[:, event[3:5], :2].numpy()
            coll_shot, miss_shot = event[2], event[5]

            fig, ax = plt.subplots(1, 2, figsize = (16, 6))
            ax[0] = self.vis_unit(ax[0], coll_tras, coll_shot, "coll_event")
            ax[1] = self.vis_unit(ax[1], miss_tras, miss_shot, "miss_event")
            plt.tight_layout()
            plt.show()
            pass
        return


    def vis_multi_ships(self):
        self.read_data()
        events = np.load(self.multi_ships_save)
        for i, event in enumerate(events):
            coll, miss = event[:6], event[7:13]
            coll = coll[coll != 0]
            miss = miss[miss != 0]
            coll_shot, miss_shot = event[6].astype(int), event[-1].astype(int)

            coll_tras = self.dataset[:, coll, :2].numpy()
            miss_tras = self.dataset[:, miss, :2].numpy()

            fig, ax = plt.subplots(1, 2, figsize = (16, 6))
            ax[0] = self.vis_unit(ax[0], coll_tras, coll_shot, "coll_event")
            ax[1] = self.vis_unit(ax[1], miss_tras, miss_shot, "miss_event")
            plt.tight_layout()
            #plt.savefig("/Users/yangkaisen/MyProject/Risk_PartC/dataset_torch/pics_risk/_1_coll.png")
            plt.show()
            pass
        return





