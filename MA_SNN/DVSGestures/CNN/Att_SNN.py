import torch
from utils import util
from DVSGestures.DVS_Gesture_utils.dataset import create_data
from DVSGestures.CNN.Networks.Att_SNN import create_net
from DVSGestures.CNN.Config import configs
from DVSGestures.DVS_Gesture_utils.process import process
from DVSGestures.DVS_Gesture_utils.save import save_csv
import time
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description='训练Attention SNN模型')
    parser.add_argument('--attention', type=str, choices=['no', 'TA', 'CA', 'SA', 'CSA', 'TSA', 'TCA', 'TCSA'], 
                        help='注意力机制选择: no, TA, CA, SA, CSA, TSA, TCA, TCSA')
    parser.add_argument('--num_epochs', type=int, 
                        help='训练轮数 (默认: 300)')
    parser.add_argument('--dt', type=int, 
                        help='时间步长 dt (默认: 25)')
    parser.add_argument('--T', type=int, 
                        help='时间窗口 T (默认: 20)')
    parser.add_argument('--batch_size', type=int, 
                        help='批次大小 (默认: 16)')
    parser.add_argument('--lr', type=float, 
                        help='学习率 (默认: 1e-4)')
    parser.add_argument('--c_ratio', type=int, 
                        help='通道比例 (默认: 8)')
    parser.add_argument('--t_ratio', type=int, 
                        help='时间比例 (默认: 5)')
    parser.add_argument('--disable_spike', action='store_true',
                        help='是否禁止发放脉冲（仅对TA有效，默认: False）')
    parser.add_argument('--result_dir', type=str, 
                        help='结果保存目录（会覆盖Config中的recordPath）')
    return parser.parse_args()

def main():

    config = configs()
    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = parse_args()
    
    # 只有当参数不为 None 时才覆盖默认值
    if args.attention is not None:
        config.attention = args.attention
        print(f"命令行设置：注意力机制={config.attention}")
    
    if args.num_epochs is not None:
        config.num_epochs = args.num_epochs
        print(f"命令行设置：训练轮数={config.num_epochs}")
    
    if args.dt is not None:
        config.dt = args.dt
        print(f"命令行设置：dt={config.dt}")
    
    if args.T is not None:
        config.T = args.T
        print(f"命令行设置：T={config.T}")
    
    if args.batch_size is not None:
        config.batch_size = args.batch_size
        print(f"命令行设置：batch_size={config.batch_size}")
    
    if args.lr is not None:
        config.lr = args.lr
        print(f"命令行设置：lr={config.lr}")
    
    if args.c_ratio is not None:
        config.c_ratio = args.c_ratio
        print(f"命令行设置：c_ratio={config.c_ratio}")
    
    if args.t_ratio is not None:
        config.t_ratio = args.t_ratio
        print(f"命令行设置：t_ratio={config.t_ratio}")
    
    if args.disable_spike:
        config.disable_spike = True
        print(f"命令行设置：disable_spike={config.disable_spike}")
    
    # 如果指定了结果目录，覆盖默认路径
    if args.result_dir is not None:
        config.recordPath = args.result_dir
        config.modelPath = args.result_dir
        if not os.path.exists(args.result_dir):
            os.makedirs(args.result_dir)
        print(f"命令行设置：结果保存目录={args.result_dir}")
    
    print(config.device)

    config.device_ids = range(torch.cuda.device_count())
    print(config.device_ids)

    config.name = (
        config.attention
        + "_SNN(CNN)-DVS-Gesture_dt="
        + str(config.dt)
        + "ms"
        + "_T="
        + str(config.T)
        + "_epoch="
        + str(config.num_epochs)
        + "_"
        + time.strftime("%Y%m%d%H%M%S", time.localtime())
    )
    config.modelNames = config.name + ".t7"
    config.recordNames = config.name + ".csv"

    print(config)

    create_data(config=config)

    create_net(config=config)

    print(config.model)

    print(util.get_parameter_number(config.model))

    process(config=config)

    print("best acc:", config.best_acc, "best_epoch:", config.best_epoch)

    save_csv(config=config)
