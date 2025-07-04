import os
import argparse
import numpy as np
from src.utils import read_dataset, read_multivariate_dataset, read_multivariate_datasetsuper

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

if __name__ == "__main__":
    # Get the arguments
    parser = argsparser()
    args = parser.parse_args()

    # Seeding
    np.random.seed(0)

    # Create output directory
    output_dir = os.path.join(output_dir, 'multivariate_datasets_' + 'supervise')
    os.makedirs(output_dir, exist_ok=True)

    # Loop through all datasets
    for dataset_name in multivariate_datasets:
        print(f"Processing dataset: {dataset_name}")

        # Read data
        X, y, train_idx, test_idx = read_multivariate_datasetsuper(multivariate_dir, dataset_name, args.shot)
        
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