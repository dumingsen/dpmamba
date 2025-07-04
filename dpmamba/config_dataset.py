def config_dataset(dataset):
    if dataset == "ArticularyWordRecognition":
        train_len = 275
        test_len = 300
        num_nodes = 9
        feature_dim = 144
        nclass = 25
    elif dataset == "AtrialFibrillation":
        train_len = 15
        test_len = 15
        num_nodes = 2
        feature_dim = 640
        nclass = 3
    elif dataset == "CharacterTrajectories":
        train_len = 1422
        test_len = 1436
        num_nodes = 3
        feature_dim = 182
        nclass = 20
    elif dataset == "FaceDetection":
        train_len = 5890
        test_len = 3524
        num_nodes = 144
        feature_dim = 62
        nclass = 2
    elif dataset == "FingerMovements":
        train_len = 316
        test_len = 100
        num_nodes = 28
        feature_dim = 50
        nclass = 2
    elif dataset == "HandMovementDirection":
        train_len = 160
        test_len = 74
        num_nodes = 10
        feature_dim = 400
        nclass = 4
    elif dataset == "Handwriting":
        train_len = 150
        test_len = 850
        num_nodes = 3
        feature_dim = 152
        nclass = 26
    elif dataset == "Heartbeat":
        train_len = 204
        test_len = 205
        num_nodes = 61
        feature_dim = 405
        nclass = 2
    elif dataset == "Libras":
        train_len = 180
        test_len = 180
        num_nodes = 2
        feature_dim = 45
        nclass = 15
    elif dataset == "LSST":
        train_len = 2459
        test_len = 2466
        num_nodes = 6
        feature_dim = 36
        nclass = 14
    elif dataset == "MotorImagery":
        train_len = 278
        test_len = 100
        num_nodes = 64
        feature_dim = 3000
        nclass = 2
    elif dataset == "NATOPS":
        train_len = 180
        test_len = 180
        num_nodes = 24
        feature_dim = 51
        nclass = 6
    elif dataset == "PEMS-SF":
        train_len = 267
        test_len = 173
        num_nodes = 963
        feature_dim = 144
        nclass = 7
    elif dataset == "PenDigits":
        train_len = 7494
        test_len = 3498
        num_nodes = 2
        feature_dim = 8
        nclass = 10
    elif dataset == "SelfRegulationSCP2":
        train_len = 200
        test_len = 180
        num_nodes = 7
        feature_dim = 1152
        nclass = 2
    elif dataset == "SpokenArabicDigits":
        train_len = 6599
        test_len = 2199
        num_nodes = 13
        feature_dim = 93
        nclass = 10
    elif dataset == "StandWalkJump":
        train_len = 12
        test_len = 15
        num_nodes = 4
        feature_dim = 2500
        nclass = 3
    else:
        raise Exception("Only support these datasets...")

    return train_len, test_len, num_nodes, feature_dim, nclass