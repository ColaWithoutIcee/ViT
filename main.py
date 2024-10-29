'''
# -*- coding: utf-8 -*-
@File    :   main.py
@Time    :   2024/10/25 15:57:52
@Author  :   Jiabing SUN 
@Version :   1.0
@Contact :   Jiabingsun777@gmail.com
@Desc    :   None
'''

# here put the import lib
import argparse
from datetime import datetime
from Solver import Solver


def main(args):
    print(args)
    solver = Solver(args)
    if args.mode == 'test':
        solver.test()
    elif args.mode == 'train':
        solver.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # exp settings
    parser.add_argument('--mode', default='train', choices=['train', 'test'], 
                        help='mode for the program')
    parser.add_argument('--model', default='vit', choices=['vit'],
                        help='models to classification')
    # model training settings
    parser.add_argument('--num_epoch', type=int, default=50, 
                        help='num of training epoch')
    parser.add_argument('--val_on_epoch', type=int, default=1,
                        help='val for each n epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                        help="batch size, 1, 2, 4, 16, ...")
    parser.add_argument('--lr', type=float, default=0.001, 
                        help="learning rate")
    parser.add_argument("--resume", type=int, default=0, choices=[0, 1],
                        help="resume training")
    # task name
    parser.add_argument("--task_name", type=str, 
                        default=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        help='your task name')
    args = parser.parse_args()
    main(args)