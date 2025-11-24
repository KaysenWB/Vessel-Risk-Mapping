import argparse
import os.path

import numpy as np
import pandas as pd

from build_dataset import show_, SamplingCollMiss
from simulate import Risk_Preception
from utils import Calculating



parser = argparse.ArgumentParser(description='make dataset')
parser.add_argument('--map_root', type=str, default='/Users/yangkaisen/MyProject/Data/map/hongkong.jpg')
parser.add_argument('--pred_dir', type=str, default='./preds')
parser.add_argument('--event_dir', type=str, default='./events')
parser.add_argument('--result_dir', type=str, default='./result')
parser.add_argument('--pics_risk_dir', type=str, default='./pics_risk')

parser.add_argument('--model', type=str, default='Seq2SeqE_32')
parser.add_argument('--coll_coef', type=float, default= 0.5)
parser.add_argument('--miss_coef', type=list, default= [3, 7.5]) #distance coef for sampling miss events (ship_size*coef)
parser.add_argument('--event_time_border', type=list, default= [48, 64])
parser.add_argument('--num_samples', type=int, default= 1200)
# [max ships, individual samples], all samples equal to above num_samples
parser.add_argument('--num_samples_multi', type=list, default= [[3, 300], [4, 300], [5, 300], [6, 300]])

parser.add_argument('--inver', type=int, default= 10, help= "second")
parser.add_argument('--alarm_adv', type=int, default= 6, help= "required adv steps of alarm")
parser.add_argument('--show', type=int, default=0, help="0, do not, 1, vis1, 2, vis2")
parser.add_argument('--extent', type=list, default=[114.099003, 114.187537, 22.265695, 22.322062])
parser.add_argument('--project_para', type=list, default=[0.0009719767418486251, 0.0008993216059187322])

parser.add_argument('--method', type=str, default="map_cri", help=["dcpa", "tcpa", "cri"," sd", "map_cri", "map_sd"," map_st"])
parser.add_argument('--threshold', type=dict, default={"cri":0.65, "sd":0.7, "sd2":0.1, "map_cri":0.3, "map_sd":0.2, "map_st":0.15})
parser.add_argument('--alarm_time', type=int, default=3) # in predicted trajs
parser.add_argument('--warn_time', type=int, default=4) # in stimulating steps
parser.add_argument('--cal_samples', type=int, default=600) # in stimulating steps
#parser.add_argument('--report_samples', type=int, default=300) # in stimulating steps
parser.add_argument('--device', type=str, default='cpu')
parser.add_argument('--d_range', type=int, default=45 * 7.5) # CRI para
parser.add_argument('--t_range', type=int, default=5) # CRI para
parser.add_argument('--grid_size', type=int, default=50) # SD para
parser.add_argument('--sd_scale', type=list, default=[7.5, 4]) # SD a and b axi
parser.add_argument('--flag', type=str, default="multi", help="pairs, multi")
parser.add_argument('--pred_type', type=str, default="multi_pred", help="single_pred, multi_pred") # SD a and b axi
parser.add_argument('--grid_mapping', type=dict, default= {"lon":150, "lat":150})
parser.add_argument('--K', type=int, default= 4)

args = parser.parse_args()


for i in [args.pred_dir, args.event_dir, args.result_dir, args.pics_risk_dir]:
    if not os.path.exists(i):
        os.makedirs(i)

Cache = SamplingCollMiss(args)
#Cache.vis_multi_ships()

#method_list = ["cri", "cri2", "sd", "sd2", "map_cri", "map_sd","map_st"]
method_list = ["cri", "cri2", "sd", "sd2"]
flags = ["pairs", "multi"]
preds_data = ["Trans_32", "Seq2SeqE_32", "TCNNe_32"]
args.pred_type = "multi_pred"


save_dir = args.result_dir
save_reports = [[] for i in range(4)]
for p in preds_data:
    for f in flags:
        for m in method_list:
            args.flag = f
            args.method = m
            args.model = p
            args.result_dir = save_dir + f"/{args.pred_type}"+ f"/{args.model}"
            if not os.path.exists(args.result_dir):
                os.makedirs(args.result_dir)

            simulator = Risk_Preception(args)
            #simulator.forward()

            best_re, whole_re = simulator.report()

            save_reports[0].append(best_re)
            save_reports[1].append(whole_re)
            save_reports[2].append(f + "_" + m)
            save_reports[3].append(args.model)

simulator.save_reports(save_reports)
