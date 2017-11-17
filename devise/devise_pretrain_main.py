import os
import argparse

import torch.nn as nn

from networks import resnet
from networks import toy_cnn
from utils.training_flow import TrainingFlow
from utils.accuracy import Accuracy


parser = argparse.ArgumentParser(description='PyTorch Devise Pre-training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to the latest checkpoint (default: none)')


if __name__ == '__main__':
    # Parameters
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    csv_log_name = 'records/log.csv'
    checkpoint_name = 'records/checkpoint.pth.tar'
    best_model_name = 'records/best_model.pth.tar'

    arch = 'ResNet18'
    optimizer_type = 'SGD'

    args = parser.parse_args()

    # Hyper-parameters
    epochs = 350
    lr = args.lr
    batch_size = 128
    saturate_patience = 40
    reduce_patience = 4

    # Initialize
    model = resnet.ResNet18()
    model = nn.DataParallel(model).cuda()

    params_to_optimize = model.parameters()
    loss_function = nn.CrossEntropyLoss()

    compute_batch_accuracy = Accuracy().compute_batch_accuracy

    training_flow = TrainingFlow(model=model,
                                 params_to_optimize=params_to_optimize,
                                 loss_function=loss_function,
                                 compute_batch_accuracy=compute_batch_accuracy,
                                 epochs=epochs,
                                 lr=lr,
                                 batch_size=batch_size,
                                 classes=classes,
                                 saturate_patience=saturate_patience,
                                 reduce_patience=reduce_patience,
                                 csv_log_name=csv_log_name,
                                 checkpoint_name=checkpoint_name,
                                 best_model_name=best_model_name,
                                 arch=arch,
                                 optimizer_type=optimizer_type,
                                 args=args)

    # Start training
    training_flow.train()
