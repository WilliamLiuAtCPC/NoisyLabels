import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.optim as optim
import os
from myCNN import CNN_NET
from myLossFunc import CE, CES_CE


os.environ["CUDA_VISIBLE_DEVICES"]='1'


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
    yita_list = [0,0.2,0.4,0.6,0.8]
    A = -4
    a = 1
    b = 1.9

    transform = transforms.Compose([transforms.Resize(40),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomVerticalFlip(),
                                    transforms.RandomCrop(32)])
    transform_test = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    testset = torchvision.datasets.CIFAR10(root='/home1/smliu1/datasets/', train=False,
                                           download=False, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=50,
                                             shuffle=False, num_workers=1)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    # os.mkdir('interest')
    name_loss = 'interest/ces1_'
    for yita in yita_list:
        trainset = torchvision.datasets.CIFAR10(root='/home1/smliu1/datasets/', train=True,
                                                download=False, transform=transform_test)
        trainset.targets = addNoise(trainset.targets, yita)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                                  shuffle=True, num_workers=1)

        result_file = open(name_loss + str(yita) + '.txt', 'w')
        net = CNN_NET()
        net = net.cuda()

        lr = 0.01
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0001)
        # loss_func = torch.nn.CrossEntropyLoss()
        # loss_func = CE()
        loss_func = CES_CE(A,a,b)
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
                bx = transform(b_x)
                outputs = net.forward(b_x.cuda())
                _, predicted = torch.max(outputs.data.cpu(), 1)
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
                    if count>100:
                        break
                    images, labels = data
                    outputs = net(images.cuda())
                    _, predicted = torch.max(outputs.data.cpu(), 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    count+=1
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
                torch.save(state, name_loss + str(yita))
                best_accuracy = test_accuracy
            if (epoch + 1) % 40 == 0:
                lr = lr * 0.1
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

        print('Finished Training')
        state = {
            'state_dict': net.state_dict()
        }
        torch.save(state, name_loss + str(yita)+'_final')
        plt.plot(epoch_record, train_perf_record, 'b')
        plt.plot(epoch_record, test_perf_record, 'r')
        plt.savefig(name_loss + str(yita) + '.jpg')
        plt.show()
        result_file.close()
