from tqdm import tqdm

import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models


class TrainingFlow():
    def __init__(self, model, epochs, lr, batch_size, classes):
        self.model = model
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.classes = classes

        self.prepare_training()

    def prepare_datasets(self):
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        self.train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        self.test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    def prepare_dataloaders(self):
        self.train_loader = torch.utils.data.DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=2)
        self.test_loader = torch.utils.data.DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=2)

    def set_loss(self):
        self.criterion = nn.CrossEntropyLoss()

    def set_optimizer(self):
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def prepare_training(self):
        self.prepare_datasets()
        self.prepare_dataloaders()
        self.set_loss()
        self.set_optimizer()

    def initialize_epoch(self):
        self.progress = tqdm(self.current_data_loader)

    def initialize_train_epoch(self):
        self.current_data_loader = self.train_loader
        self.initialize_epoch()
        self.model.train() # switch to train mode

        print '-' * 60
        print 'Training stage, epoch:', self.epoch + 1
        print '-' * 60

    def initialize_val_epoch(self):
        self.current_data_loader = self.test_loader
        self.initialize_epoch()
        self.model.eval() # switch to evaluate mode

        print '-' * 60
        print 'Validation stage, epoch:', self.epoch + 1
        print '-' * 60

    def train_one_epoch(self):
        self.initialize_train_epoch()
        print_steps = 1000
        running_loss = 0.0
        for self.iteration_count, data in enumerate(self.progress):
            inputs, labels = data  # get the inputs
            inputs, labels = Variable(inputs), Variable(labels)  # wrap them in Variable

            self.optimizer.zero_grad()  # zero the parameter gradients

            # forward + backward + optimize
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            # print statistics
            running_loss += loss.data[0]
            if self.iteration_count % print_steps == print_steps - 1:  # print every print_steps mini-batches
                print('[%d, %5d] loss: %.3f' % (self.epoch + 1, self.iteration_count + 1, running_loss / print_steps))
                running_loss = 0.0

    def validate_one_epoch(self):
        self.initialize_val_epoch()

        correct = 0
        total = 0
        for self.iteration_count, data in enumerate(self.progress):
            images, labels = data
            outputs = self.model(Variable(images))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()

        self.val_acc = 100 * correct / total
        print '*' * 60
        print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
        print '*' * 60

    def train(self):
        self.best_val_acc = 0.
        self.patience = 5
        self.saturate_count = 0
        for self.epoch in range(self.epochs):
            self.train_one_epoch()
            self.validate_one_epoch()

            if self.val_acc > self.best_val_acc:
                self.best_val_acc = self.val_acc
                self.saturate_count = 0
            else:
                self.saturate_count += 1
                if self.saturate_count >= self.patience:
                    break

        print('Finished Training')
