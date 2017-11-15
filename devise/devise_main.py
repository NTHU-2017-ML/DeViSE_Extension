import os

import torch.nn as nn

from networks import resnet
from networks import toy_cnn
from utils.training_flow import TrainingFlow


if __name__ == '__main__':
    # Parameters
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    csv_log_name = 'records/log.csv'
    checkpoint_name = 'records/checkpoint.pth.tar'
    best_model_name = 'records/best_model.pth.tar'

    arch = 'ResNet18'

    # Hyper-parameters
    epochs = 50
    lr = 1e-5
    batch_size = 64
    patience = 10

    # Initialize
    model = resnet.ResNet18()
    #model = toy_cnn.ToyCnn()
    model = nn.DataParallel(model).cuda()
    training_flow = TrainingFlow(model, epochs, lr, batch_size, classes,
                                 patience=patience,
                                 csv_log_name=csv_log_name,
                                 checkpoint_name=checkpoint_name,
                                 best_model_name=best_model_name,
                                 arch=arch)

    # Start training
    training_flow.train()
