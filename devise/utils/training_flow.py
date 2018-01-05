from tqdm import tqdm
import shutil
import pandas as pd
import os

import torch
from torch.optim import Adam, SGD, lr_scheduler
import torch.nn as nn
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models


class TrainingFlow():
    def __init__(self, model=None, params_to_optimize=None, loss_function=None, compute_batch_accuracy=None, epochs=200, lr=0.1, batch_size=32, classes=None,
                 saturate_patience=20, reduce_patience=5, cooldown=12,
                 csv_log_name='', checkpoint_name='', best_model_name='', arch='', optimizer_type='Adam', args=None):
        self.model = model
        self.params_to_optimize = params_to_optimize
        self.loss_function = loss_function
        self.compute_batch_accuracy = compute_batch_accuracy
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.classes = classes
        self.saturate_patience = saturate_patience
        self.reduce_patience = reduce_patience
        self.cooldown = cooldown
        self.csv_log_name = csv_log_name
        self.checkpoint_name = checkpoint_name
        self.best_model_name = best_model_name
        self.arch = arch
        self.args = args
        self.optimizer_type = optimizer_type

        self.start_epoch = 1
        self.best_val_acc = 0.
        self.saturate_count = 0

        self.prepare_training()

    def prepare_datasets(self):
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        self.train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        self.test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    def prepare_dataloaders(self):
        self.train_loader = torch.utils.data.DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=2)
        self.test_loader = torch.utils.data.DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=2)

    def set_loss(self):
        self.criterion = self.loss_function

    def set_optimizer(self):
        if self.optimizer_type == 'SGD':
            self.optimizer = SGD(self.params_to_optimize, lr=self.lr, momentum=0.9, weight_decay=5e-4)
        else:
            self.optimizer = Adam(self.params_to_optimize, lr=self.lr)

    def set_scheduler(self):
        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, 'max', patience=self.reduce_patience, cooldown=self.cooldown, verbose=True)

    def resume(self):
        args = self.args
        if args.resume:
            if os.path.isfile(args.resume):
                print(("=> loading checkpoint '{}'".format(args.resume)))
                checkpoint = torch.load(args.resume)
                self.start_epoch = checkpoint['epoch'] + 1
                self.best_val_acc = checkpoint['best_val_acc']
                self.model.load_state_dict(checkpoint['state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                print(("=> loaded checkpoint '{}' (epoch {})".format(args.resume, self.start_epoch)))
            else:
                print(("=> no checkpoint found at '{}'".format(args.resume)))

    def prepare_training(self):
        self.prepare_datasets()
        self.prepare_dataloaders()
        self.set_loss()
        self.set_optimizer()
        self.set_scheduler()
        self.resume()

    def initialize_epoch(self):
        self.progress = tqdm(self.current_data_loader)

    def initialize_train_epoch(self):
        self.current_data_loader = self.train_loader
        self.train_epoch_acc = 0.0
        self.running_loss = 0.0
        self.train_epoch_loss = 0.0
        self.initialize_epoch()
        self.model.train() # switch to train mode

        print('-' * 80, '\n', '-' * 80, '\n', '-' * 80)
        print('Training stage, epoch:', self.epoch)
        print('-' * 80, '\n', '-' * 80, '\n', '-' * 80)

    def initialize_val_epoch(self):
        self.current_data_loader = self.test_loader
        self.initialize_epoch()
        self.model.eval() # switch to evaluate mode

        print('-' * 80, '\n', '-' * 80, '\n', '-' * 80)
        print('Validation stage, epoch:', self.epoch)
        print('-' * 80, '\n', '-' * 80, '\n', '-' * 80)

    def print_train_batch_statistics(self):
        self.running_loss += self.loss.data[0]
        if self.iteration_count % self.print_steps == self.print_steps - 1:  # print every print_steps mini-batches
            print(('[%d, %5d] loss: %.3f' % (self.epoch, self.iteration_count + 1, self.running_loss / self.print_steps)))
            self.running_loss = 0.0

    def print_train_epoch_statistics(self):
        print('*' * 60, '\n', '*' * 60)
        print(('Training accuracy of this epoch: %.1f %%' % self.train_epoch_acc))
        print(('Training loss of this epoch: %.3f' % self.train_epoch_loss))
        print('*' * 60, '\n', '*' * 60, '\n')

    def print_val_statistics(self):
        print('*' * 60, '\n', '*' * 60)
        print(('Validation accuracy of this epoch: %.1f %%' % self.val_acc))
        print('*' * 60, '\n', '*' * 60, '\n')

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
        column_names = ['Epoch', 'Arch', 'Optimizer-type', 'Learning-rate', 'Batch-size', 'Saturate-patience', 'Cooldown', 'Train-Loss', 'Train-Acc', 'Val-Acc']
        info_dict = {column_names[0]: [self.epoch],
                     column_names[1]: [self.arch],
                     column_names[2]: [str(type(self.optimizer))],
                     column_names[3]: [self.optimizer.param_groups[0]['lr']],
                     column_names[4]: [self.batch_size],
                     column_names[5]: [self.saturate_patience],
                     column_names[6]: [self.cooldown],
                     column_names[7]: [round(self.train_epoch_loss, 3)],
                     column_names[8]: [round(self.train_epoch_acc, 3)],
                     column_names[9]: [round(self.val_acc, 3)]}

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
                 'best_val_acc': self.best_val_acc,
                 'optimizer' : self.optimizer.state_dict()}
        torch.save(state, checkpoint_name)

        if self.is_best:
            shutil.copyfile(checkpoint_name, self.best_model_name)

    def check_saturate(self):
        is_saturate = False
        if self.is_best:
            self.best_val_acc = self.val_acc
            self.saturate_count = 0
        else:
            self.saturate_count += 1
            if self.saturate_count >= self.saturate_patience:
                is_saturate = True
        self.is_saturate = is_saturate

    def train(self):
        for self.epoch in range(self.start_epoch, self.epochs + 1):
            self.train_one_epoch()
            self.validate_one_epoch()
            self.scheduler.step(self.val_acc) # call lr_scheduler

            self.write_csv_logs()

            self.is_best = self.val_acc > self.best_val_acc
            self.check_saturate()
            self.save_checkpoints()
            if self.is_saturate:
                print('Validation accuracy is saturate!')
                break

        print('Finished Training')
