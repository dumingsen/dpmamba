import os
import argparse

import numpy as np
import torch

from src.utils import read_dataset_from_npy, Logger
from src.simtsc.model import SimTSC, SimTSCTrainer
from print import print
from printt import printt
data_dir = r'/root/SimTSC-main/SimTSC-main/tmp' # H:\0819\0816 G:\桌面\0816
log_dir =  r'/root/SimTSC-main/SimTSC-main/logs'

#multivariate_datasets = ['CharacterTrajectories', 'ECG', 'KickvsPunch', 'NetFlow']
multivariate_datasets = ["AtrialFibrillation", "FingerMovements", "HandMovementDirection", "Heartbeat",
                "Libras","MotorImagery","NATOPS","SelfRegulationSCP2","StandWalkJump",'CharacterTrajectories',"PenDigits"]

def train(X, y, train_idx, test_idx, distances, device, logger, K, alpha, args1, args):
    print('X', X.shape)#(2858, 3, 205)
    nb_classes = len(np.unique(y, axis=0))
    print('textidx', len(test_idx))#572
    input_size = X.shape[1]

    model = SimTSC(input_size, nb_classes, args1)
    model = model.to(device)
    trainer = SimTSCTrainer(device, logger, args)

    check_idx_overlap(train_idx, test_idx)
    model = trainer.fit(model, X, y, train_idx, distances, K, alpha,test_idx)
    
    acc = trainer.testo(model, test_idx)

    return acc
def check_idx_overlap(train_idx, test_idx):
    overlap = np.intersect1d(train_idx, test_idx)
    if len(overlap) > 0:
        printt(f"Warning: train_idx and test_idx have {len(overlap)} overlapping indices.")
    else:
        printt("train_idx and test_idx are distinct.")


def argsparser():
    parser = argparse.ArgumentParser("SimTSC")
    parser.add_argument('--dataset', help='Dataset name', default='FingerMovements')
    #coffee
    #  CharacterTrajectories 
    # AtrialFibrillation#0.6667 0.500 
    #  FingerMovements 0.6310
    #HandMovementDirection 0.319 0.340
    #Heartbeat 0.6585
    #Libras 有问题 准确率过低
    #MotorImagery 0.61
    #'NATOPS' 0.5556
    #"SelfRegulationSCP2" 0.53
    #"StandWalkJump" 0.667
    #"PenDigits"

    parser.add_argument('--seed', help='Random seed', type=int, default=0)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--shot', help='shot', type=int, default=1)#标记数量 1
    parser.add_argument('--K', help='K', type=int, default=3)#3
    parser.add_argument('--alpha', help='alpha', type=float, default=0.3)

    return parser
def argsparser1():
    parser = argparse.ArgumentParser(description='iTransformer')

    # basic config
    parser.add_argument('--is_training', type=int, required=False, default=1, help='status')#
    parser.add_argument('--model_id', type=str, required=False, default='test', help='model id')#
    parser.add_argument('--model', type=str, required=False, default='S_Mamba',
                        help='model name, options: [iTransformer, iInformer, iReformer, iFlowformer, iFlashformer,S_Mamba ]')

    # data loader
    parser.add_argument('--data', type=str, required=False, default='custom', help='dataset type')#

    parser.add_argument('--root_path', type=str, default=r'G:\桌面\0816\S-D-Mamba-main\S-D-Mamba-main\data\electricity', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='electricity.csv', help='data csv file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    # a=205
    #parser.add_argument('--seq_len', type=int, default=205, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length') # no longer needed in inverted Transformers
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')

    # model define
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size') # applicable on arbitrary number of variates in inverted Transformers
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')##128 512
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=1, help='num of encoder layers')#2
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=512, help='dimension of fcn')#2048
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=2, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    # iTransformer
    parser.add_argument('--exp_name', type=str, required=False, default='MTSF',
                        help='experiemnt name, options:[MTSF, partial_train]')
    parser.add_argument('--channel_independence', type=bool, default=False, help='whether to use channel_independence mechanism')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)
    parser.add_argument('--class_strategy', type=str, default='projection', help='projection/average/cls_token')
    parser.add_argument('--target_root_path', type=str, default='./data/electricity/', help='root path of the data file')
    parser.add_argument('--target_data_path', type=str, default='electricity.csv', help='data file')
    parser.add_argument('--efficient_training', type=bool, default=False, help='whether to use efficient_training (exp_name should be partial train)') # See Figure 8 of our paper for the detail
    parser.add_argument('--use_norm', type=int, default=True, help='use norm and denorm')
    parser.add_argument('--partial_start_index', type=int, default=0, help='the start index of variates for partial training, '
                                                                           'you can select [partial_start_index, min(enc_in + partial_start_index, N)]')
    parser.add_argument('--d_state', type=int, default=32, help='parameter of Mamba Block')

    args1 = parser.parse_args()
    args1.use_gpu = True if torch.cuda.is_available() and args1.use_gpu else False

    if args1.use_gpu and args1.use_multi_gpu:
        args1.devices = args1.devices.replace(' ', '')
        device_ids = args1.devices.split(',')
        args1.device_ids = [int(id_) for id_ in device_ids]
        args1.gpu = args1.device_ids[0]

    print('Args in experiment:')
    print(args1)
    return args1

def normalize_dtw_matrix(dtw_matrix, method='normalize'):
    """
    对 DTW 相似矩阵进行归一化或标准化。
    
    参数：
    dtw_matrix: 输入的 DTW 相似矩阵
    method: 'normalize' 进行归一化，'standardize' 进行标准化
    
    返回：
    处理后的矩阵
    """
    if method == 'normalize':
        # 归一化到 [0, 1]
        min_val = np.min(dtw_matrix)
        max_val = np.max(dtw_matrix)
        normalized_matrix = (dtw_matrix - min_val) / (max_val - min_val)
        return normalized_matrix
    
    elif method == 'standardize':
        # 标准化到均值为 0，标准差为 1
        mean = np.mean(dtw_matrix)
        std = np.std(dtw_matrix)
        standardized_matrix = (dtw_matrix - mean) / std
        return standardized_matrix
    
    else:
        raise ValueError("Method must be 'normalize' or 'standardize'.")

if __name__ == "__main__":
    # Get the arguments
    parser = argsparser()
    args = parser.parse_args()
    args1 = argsparser1()
    # parser1 = argsparser1()
    # args1 = parser1.parse_args()

    

    # Setup the gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        printt("--> Running on the GPU")
    else:
        device = torch.device("cpu")
        printt("--> Running on the CPU")

    # Seeding
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ##获得dtw
    if args.dataset in multivariate_datasets:
        dtw_dir = os.path.join(r'/root/SimTSC-main/SimTSC-main/tmp/multivariate_datasets_dtw') # H:\0819\0816 G:\桌面\0816
        distances = np.load(os.path.join(dtw_dir, args.dataset+'.npy'))#_dtw  # +'_allone'
    else:
        dtw_dir = os.path.join(data_dir, 'ucr_datasets_dtw') 
        distances = np.load(os.path.join(dtw_dir, args.dataset+'.npy'))
    
    out_dir = os.path.join(log_dir, 'simtsc_log_'+str(args.shot)+'_shot'+str(args.K)+'_'+str(args.alpha))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_path = os.path.join(log_dir, args.dataset+'_'+str(args.seed)+'.txt')

    with open(out_path, 'w') as f:
        logger = Logger(f)

        #===============================
        args.sepervise=False#True
        supervise=args.sepervise
        args.flexible=True
        args.bili=0.2
        #===========================
        # 获得Read data
        if args.dataset in multivariate_datasets:
            if not supervise:
                if not args.flexible:
                    printt('固定标记样本数量！！！！！！！！！！！')
                    X, y, train_idx, test_idx = read_dataset_from_npy(os.path.join(data_dir, 'multivariate_datasets_'+str(args.shot)+'_shot', args.dataset+'.npy'))
                else:
                    printt('按比例得标记样本数量！！！！！！！！')
                    X, y, train_idx, test_idx = read_dataset_from_npy(os.path.join(data_dir, 'multivariate_datasets_' + 'flexible_' +str(args.bili)+ '_shot', args.dataset+'.npy'))
            else:
                printt('监督版本！！！！！！')
                X, y, train_idx, test_idx = read_dataset_from_npy(os.path.join(data_dir, 'multivariate_datasets_' + 'supervise', args.dataset+'.npy'))
                
        else:
            X, y, train_idx, test_idx = read_dataset_from_npy(os.path.join(data_dir, 'ucr_datasets_datasets_'+str(args.shot)+'_shot', args.dataset+'.npy'))

        # Train the model
        printt('标记数量',len(train_idx))
        printt('X, y, train_idx, test_idx, distances, device, logger, args.K, args.alpha',
              X.shape, y.shape, len(train_idx), len(test_idx), distances.shape, device, logger, args.K, args.alpha)
        #input()

        a=640
        args1.seq_len= X.shape[1]
        printt('X.shape[1]',X.shape[1])

        #input()
        distances=normalize_dtw_matrix(distances, method='normalize')
        acc = train(X, y, train_idx, test_idx, distances, device, logger, args.K, args.alpha,args1, args)
        #(56, 1, 286) (56,) [33, 45] [23 36 21 19  9 39 51  3  0 53 47 44] 
        # # (56, 56) cpu <src.utils.Logger object at 0x000001F324CAEEF0> 3 0.3
        logger.log('--> {} Test Accuracy: {:5.4f}'.format(args.dataset, acc))
        logger.log(str(acc))
