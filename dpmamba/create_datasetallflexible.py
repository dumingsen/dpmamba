import math
import os
import argparse
import numpy as np
from src.utils import read_dataset, read_multivariate_dataset

dataset_dir = './datasets/UCRArchive_2018'
multivariate_dir = r'H:\0819\0816\SimTSC-main\SimTSC-main\datasets\multivariate'

output_dir = r'/root/SimTSC-main/SimTSC-main/tmp'

multivariate_datasets = [
    "AtrialFibrillation", "FingerMovements", "PenDigits", 
    "HandMovementDirection", "Heartbeat", "Libras",
    "MotorImagery", "NATOPS", "SelfRegulationSCP2", "StandWalkJump"
]
#multivariate_datasets = ["Libras"]
def argsparser():
    parser = argparse.ArgumentParser("SimTSC data creator")
    parser.add_argument('--shot', help='How many labeled time-series per class', type=int, default=1)
    return parser
from  config_dataset import config_dataset
if __name__ == "__main__":
    # Get the arguments
    parser = argsparser()
    args = parser.parse_args()
    args.bili=0.1#0.05

    # Seeding
    np.random.seed(0)

    # Create output directory

    #版本1 ，固定数量
    #output_dir = os.path.join(output_dir, 'multivariate_datasets_' + str(args.shot) + '_shot')#
    
    #版本2，固定比例
    output_dir = os.path.join(output_dir, 'multivariate_datasets_' + 'flexible_' +str(args.bili)+ '_shot')#
    os.makedirs(output_dir, exist_ok=True)

    

    # Loop through all datasets
    for dataset_name in multivariate_datasets:

        train_len, test_len, _, _, nclass=config_dataset(dataset_name)
        num=train_len+test_len
        train_lenn = int(num * 0.8)  # 训练集数量
        test_lenn = int(num * 0.2)    # 测试集数量

        # 计算训练集中标记样本的数量
        
        labeled_count = int(train_lenn * args.bili)  # 标记样本数量

        # 计算每个类别的标记样本数量，上
        labeled_per_class = labeled_count / nclass
        labeled_per_class=math.ceil(labeled_per_class)

        # 向下取整
        #labeled_per_class_floor = math.floor(labeled_per_class)

        # 四舍五入
        #labeled_per_class_rounded = round(labeled_per_class)

        print(f"Processing dataset: {dataset_name}")

        # Read data
        X, y, train_idx, test_idx = read_multivariate_dataset(multivariate_dir, dataset_name, labeled_per_class)
        
        print('X, y, train_idx, test_idx:', X.shape, y.shape, len(train_idx), len(test_idx))
        
        train_idx = np.array(train_idx)  # 确保是 NumPy 数组
        test_idx = np.array(test_idx)
        
        data = {
            'X': X,
            'y': y,
            'train_idx': train_idx,
            'test_idx': test_idx
        }
        
        # Save as .npy file
        np.save(os.path.join(output_dir, f"{dataset_name}.npy"), data)
        print(f"Data saved to {os.path.join(output_dir, f'{dataset_name}.npy')}")