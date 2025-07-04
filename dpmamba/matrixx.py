import numpy as np

# 生成一个边长为 10992 的全 1 矩阵
matrix_size = 10992
matrix = np.ones((matrix_size, matrix_size))  # 生成全 1 矩阵 np.random.rand np.ones
# 将对角线的值设为 0
np.fill_diagonal(matrix, 0)
a=["AtrialFibrillation", "FingerMovements", "HandMovementDirection", "Heartbeat",
                "Libras","MotorImagery","NATOPS","SelfRegulationSCP2","StandWalkJump",'CharacterTrajectories',"PenDigits"]

# 指定保存路径和文件名
save_path = '/root/SimTSC-main/SimTSC-main/tmp/multivariate_datasets_dtw/PenDigits_allone.npy'

# 保存矩阵到 .npy 文件
np.save(save_path, matrix)

print("矩阵已保存到:", save_path)

matrix = np.load(save_path)

# 输出矩阵形状
print("矩阵形状:", matrix)