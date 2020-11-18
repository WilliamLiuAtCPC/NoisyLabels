import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
# import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import os
from myLossFunc import CE, CES_CE2


os.environ["CUDA_VISIBLE_DEVICES"]='1'


class CNN_NET(nn.Module):
    def __init__(self):
        super().__init__()
        '''输入为3*32*32，尺寸减半是因为池化层'''
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )  # 输出为32*32*64
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )  # 输出为16*16*128
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 196, 3, stride=1, padding=1),
            nn.BatchNorm2d(196),
            nn.ReLU(),
            nn.Conv2d(196, 196, 3, stride=1, padding=1),
            nn.BatchNorm2d(196),
            nn.ReLU()
        )  # 输出为8*8*196
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Sequential(
            nn.Linear(3136, 256),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        self.fc2 = nn.Linear(256, 10)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        # x = self.dropout(x)
        x = self.fc1(x)
        # x = self.dropout(x)
        x = self.fc2(x)
        # x = nn.Softmax(x)
        return x


def getNumCorrect(output, target):
    num_correct = 0
    return num_correct


def addNoise(label_array, yita):
    a = np.arange(10)
    for i in range(len(label_array)):
        p = np.empty(10)
        p.fill(yita / 9)
        p[label_array[i]] = 1 - yita
        label_array[i] = np.random.choice(a=a, p=p)
    # print(label_array)
    return label_array


if __name__ == '__main__':
    BATCH_SIZE = 64
    EPOCH = 120
    yita = 0.8
    A = -4
    a_list = [0.001,0]
    b_list = [0.0001,0.001,0.01,10,100]
    a = 0.1
    b=0.1
    name_dir = 'ab_change_big0.8'
    # os.mkdir(name_dir)
    name_loss = name_dir+'/ce_ces_'

    transform = transforms.Compose([transforms.Resize(40),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomVerticalFlip(),
                                    transforms.RandomCrop(32)])
    transform_test = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.CIFAR10(root='/home1/smliu1/datasets/', train=True,
                                            download=False, transform=transform_test)
    trainset.targets = addNoise(trainset.targets, yita)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                              shuffle=True, num_workers=1)
    testset = torchvision.datasets.CIFAR10(root='/home1/smliu1/datasets/', train=False,
                                           download=False, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                             shuffle=False, num_workers=1)
    # l_a = [ 0.16, 0.17, 0.18, 0.19]
    # l_a = [0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18]
    l_a = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    # l_a = [0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.11,0.12,0.13,0.14,0.15,0.16,0.17,0.18,0.19]
    # l_b = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9]
    l_b = [0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8]
    grid_file = open(name_dir+'/ab_change.txt','w')
    record = 42 #189
    num = 0
    for a in l_a:
        for b in l_b:
            if num<record:
                num+=1
                continue
            print('case '+str(num))
            result_file = open(name_loss + str(a) +'_'+str(b)+ '.txt', 'w')
            net = CNN_NET()
            net = net.cuda()

            lr = 0.01
            optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0001)
            # loss_func = torch.nn.CrossEntropyLoss()
            loss_func = CES_CE2(A,a,b)
            # loss_func = CE()
            train_perf_record = []
            test_perf_record = []
            epoch_record = []

            best_accuracy = 0

            for epoch in range(EPOCH):
                running_loss = 0.0
                correct = 0
                total = 0
                for step, data in enumerate(trainloader):
                    b_x, b_y = data
                    # b_x, b_y = b_x.cuda(), b_y.cuda()
                    b_x = b_x.cuda()
                    bx = transform(b_x)
                    # b_y = addNoise(b_y, yita)
                    outputs = net.forward(b_x)
                    outputs = outputs.cpu()
                    _, predicted = torch.max(outputs.data, 1)
                    total += b_y.size(0)
                    correct += (predicted == b_y).sum().item()
                    loss = loss_func(outputs, b_y)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                train_accuracy = 100 * correct / total

                correct = 0
                total = 0
                count = 0
                with torch.no_grad():
                    for data in testloader:
                        if count > 100:
                            break
                        images, labels = data
                        outputs = net(images.cuda())
                        _, predicted = torch.max(outputs.data.cpu(), 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                        count += 1
                test_accuracy = 100 * correct / total
                tmp_str = '%d,%.3f,%.3f' % (epoch, train_accuracy, test_accuracy)
                print('epoch %d: loss= %.3f, train_perf= %.3f %%, test_perf= %.3f %%' % (
                epoch, running_loss / 50000, train_accuracy, test_accuracy))
                result_file.write(tmp_str + '\n')
                train_perf_record.append(train_accuracy)
                test_perf_record.append(test_accuracy)
                epoch_record.append(epoch)
                if test_accuracy >= best_accuracy:
                    state = {
                        'state_dict': net.state_dict()
                    }
                    torch.save(state, name_loss + str(a)+'_'+str(b))
                    best_accuracy = test_accuracy
                elif test_accuracy<best_accuracy*0.9:
                    break
                if (epoch + 1) % 40 == 0:
                    lr = lr * 0.1
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr
            print('Finished Training')
            plt.plot(epoch_record, train_perf_record, 'b')
            plt.plot(epoch_record, test_perf_record, 'r')
            plt.savefig(name_loss + str(a)+'_'+str(b) + '.jpg')
            plt.show()
            result_file.close()
            checkpoint = torch.load(name_loss + str(a)+'_'+str(b))
            net.load_state_dict(checkpoint['state_dict'])
            total = 0
            correct = 0
            count = 0
            with torch.no_grad():
                for data in testloader:
                    if count <= 100:
                        count += 1
                    else:
                        images, labels = data
                        outputs = net(images.cuda())
                        _, predicted = torch.max(outputs.data.cpu(), 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
            test_accuracy = 100 * correct / total
            grid_file.write(str(a)+','+str(b)+','+str(test_accuracy)+'\n')
            num+=1
    grid_file.close()