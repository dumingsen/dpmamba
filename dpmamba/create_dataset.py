import os
import argparse

import numpy as np

from src.utils import read_dataset, read_multivariate_dataset

dataset_dir = './datasets/UCRArchive_2018'
multivariate_dir = 'H:\0819\0816\SimTSC-main\SimTSC-main\datasets\multivariate'
output_dir = r'/root/SimTSC-main/SimTSC-main/tmp'#有用
a = '/root/MF-Net-1202/data/Multivariate_arff'#没用

multivariate_datasets = ["AtrialFibrillation", "FingerMovements","PenDigits", "HandMovementDirection", "Heartbeat",
                "Libras","MotorImagery","NATOPS","SelfRegulationSCP2","StandWalkJump"]#
#['CharacterTrajectories', 'ECG', 'KickvsPunch', 'NetFlow']

def argsparser():
    parser = argparse.ArgumentParser("SimTSC data creator")
    parser.add_argument('--dataset', help='Dataset name', default='PenDigits')
    parser.add_argument('--seed', help='Random seed', type=int, default=0)

    parser.add_argument('--shot', help='How many labeled time-series per class', type=int, default=20)#1

    return parser

if __name__ == "__main__":
    # Get the arguments
    parser = argsparser()
    args = parser.parse_args()

    # Seeding
    np.random.seed(args.seed)

    # Create dirs
    if args.dataset in multivariate_datasets:
        output_dir = os.path.join(output_dir, 'multivariate_datasets_'+str(args.shot)+'_shot')
    else:
        output_dir = os.path.join(output_dir, 'ucr_datasets_'+str(args.shot)+'_shot')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Read data
    if args.dataset in multivariate_datasets:
        X, y, train_idx, test_idx = read_multivariate_dataset(multivariate_dir, args.dataset, args.shot)
    else:
        X, y, train_idx, test_idx = read_dataset(dataset_dir, args.dataset, args.shot)
    print('X, y, train_idx, test_idx',X.shape, y.shape, len(train_idx), len(test_idx))#(2858, 3, 205) (2858,) 20 572
    train_idx = np.array(train_idx)  # 确保是 NumPy 数组
    test_idx = np.array(test_idx)
    data = {
                'X': X,
                'y': y,
                'train_idx': train_idx,
                'test_idx': test_idx
            }
    # dataset_name = f"{args.dataset}.npy"  # 假设 args.dataset 是一个字符串
    # np.save(os.path.join(output_dir, dataset_name), data)
    np.save(os.path.join(output_dir, args.dataset), data)
