from tqdm import tqdm
import shutil
import pandas as pd
import os

import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models


class TrainingFlow():
    def __init__(self, model, epochs, lr, batch_size, classes, patience=5, csv_log_name='', checkpoint_name='', best_model_name='', arch=''):
        self.model = model
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.classes = classes
        self.patience = patience
        self.csv_log_name = csv_log_name
        self.checkpoint_name = checkpoint_name
        self.best_model_name = best_model_name
        self.arch = arch

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
        self.train_epoch_acc = 0.0
        self.running_loss = 0.0
        self.train_epoch_loss = 0.0
        self.initialize_epoch()
        self.model.train() # switch to train mode

        print '-' * 80, '\n', '-' * 80, '\n', '-' * 80
        print 'Training stage, epoch:', self.epoch
        print '-' * 80, '\n', '-' * 80, '\n', '-' * 80

    def initialize_val_epoch(self):
        self.current_data_loader = self.test_loader
        self.initialize_epoch()
        self.model.eval() # switch to evaluate mode

        print '-' * 80, '\n', '-' * 80, '\n', '-' * 80
        print 'Validation stage, epoch:', self.epoch
        print '-' * 80, '\n', '-' * 80, '\n', '-' * 80

    def compute_batch_accuracy(self, outputs, targets):
        predictions = outputs.max(dim=1)[1]
        correct = ((predictions == targets).sum()).data[0]
        total = targets.size(0)
        accuracy = float(correct) / float(total)
        return correct, accuracy

    def print_train_batch_statistics(self):
        self.running_loss += self.loss.data[0]
        if self.iteration_count % self.print_steps == self.print_steps - 1:  # print every print_steps mini-batches
            print('[%d, %5d] loss: %.3f' % (self.epoch, self.iteration_count + 1, self.running_loss / self.print_steps))
            self.running_loss = 0.0

    def print_train_epoch_statistics(self):
        print '*' * 60, '\n', '*' * 60
        print('Training accuracy of this epoch: %.1f %%' % self.train_epoch_acc)
        print('Training loss of this epoch: %.3f' % self.train_epoch_loss)
        print '*' * 60, '\n', '*' * 60, '\n'

    def print_val_statistics(self):
        print '*' * 60, '\n', '*' * 60
        print('Validation accuracy of this epoch: %.1f %%' % self.val_acc)
        print '*' * 60, '\n', '*' * 60, '\n'

    def train_one_epoch(self):
        self.initialize_train_epoch()
        self.print_steps = len(self.train_loader) / 10
        for self.iteration_count, data in enumerate(self.progress, 0):
            inputs, labels = data  # get the inputs
            inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()  # wrap them in Variable and move to GPU

            self.optimizer.zero_grad()  # zero the parameter gradients

            # forward + backward + optimize
            outputs = self.model(inputs)
            self.loss = self.criterion(outputs, labels)
            self.loss.backward()
            self.optimizer.step()

            # statistics
            _, train_batch_acc = self.compute_batch_accuracy(outputs, labels)
            self.train_epoch_acc += train_batch_acc
            self.train_epoch_loss += self.loss.data[0]
            self.print_train_batch_statistics()

        iterations = self.iteration_count + 1
        self.train_epoch_acc = 100 * self.train_epoch_acc / iterations
        self.train_epoch_loss = self.train_epoch_loss / iterations
        self.print_train_epoch_statistics()

    def validate_one_epoch(self):
        self.initialize_val_epoch()

        correct = 0
        total = 0
        for self.iteration_count, data in enumerate(self.progress, 0):
            images, labels = data
            images, labels = Variable(images).cuda(), Variable(labels).cuda()  # wrap them in Variable and move to GPU
            outputs = self.model(images)

            batch_correct, _ = self.compute_batch_accuracy(outputs, labels)
            correct += batch_correct
            total += labels.size(0)

        self.val_acc = 100 * correct / total
        self.print_val_statistics()

    def write_csv_logs(self):
        column_names = ['Epoch', 'Arch', 'Learning-rate', 'Batch-size', 'Patience', 'Train-Loss', 'Train-Acc', 'Val-Acc']
        info_dict = {column_names[0]: [self.epoch],
                     column_names[1]: [self.arch],
                     column_names[2]: [self.lr],
                     column_names[3]: [self.batch_size],
                     column_names[4]: [self.patience],
                     column_names[5]: [round(self.train_epoch_loss, 3)],
                     column_names[6]: [round(self.train_epoch_acc, 3)],
                     column_names[7]: [round(self.val_acc, 3)]}

        csv_log_name = self.csv_log_name
        data_frame = pd.DataFrame.from_dict(info_dict)
        if not os.path.isfile(csv_log_name):
            data_frame.to_csv(csv_log_name, index=False, columns=column_names)
        else: # else it exists so append without writing the header
            data_frame.to_csv(csv_log_name, mode='a', header=False, index=False, columns=column_names)

    def save_checkpoints(self):
        checkpoint_name = self.checkpoint_name
        state = {'epoch': self.epoch,
                 'arch': self.arch,
                 'dataset': 'CIFAR10',
                 'state_dict': self.model.state_dict(),
                 'val_acc': self.val_acc,
                 'optimizer' : self.optimizer.state_dict()}
        torch.save(state, checkpoint_name)

        if self.is_best:
            shutil.copyfile(checkpoint_name, self.best_model_name)

    def is_saturate(self):
        is_saturate = False
        if self.is_best:
            self.best_val_acc = self.val_acc
            self.saturate_count = 0
        else:
            self.saturate_count += 1
            if self.saturate_count >= self.patience:
                is_saturate = True
        return is_saturate

    def train(self):
        self.best_val_acc = 0.
        self.saturate_count = 0
        for self.epoch, _ in enumerate(range(self.epochs), 1):
            self.train_one_epoch()
            self.validate_one_epoch()

            self.write_csv_logs()

            self.is_best = self.val_acc > self.best_val_acc
            self.save_checkpoints()
            if self.is_saturate():
                print 'Validation accuracy is saturate!'
                break

        print('Finished Training')
