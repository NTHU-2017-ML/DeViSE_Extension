import os

from networks import resnet
from networks import toy_cnn
from utils.training_flow import TrainingFlow


if __name__ == '__main__':
    # Parameters
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # Hyper-parameters
    epochs = 50
    lr = 1e-4
    batch_size = 4

    # Initialize
    #model = resnet.ResNet18()
    model = toy_cnn.ToyCnn()
    training_flow = TrainingFlow(model, epochs, lr, batch_size, classes)

    # Start training
    training_flow.train()
