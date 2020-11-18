import torch
import torchvision
import torchvision.transforms as transforms


from myCNN import CNN_NET

if __name__ == '__main__':
    BATCH_SIZE = 50
    transform_test = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    testset = torchvision.datasets.CIFAR10(root='/home1/smliu1/datasets/', train=False,
                                           download=False, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                             shuffle=False, num_workers=1)
    checkpoint = torch.load('ce_0')#load in the saved parameters
    net = CNN_NET()
    net.load_state_dict(checkpoint['state_dict'])
    net = net.cuda()
    total=0
    correct = 0
    count = 0
    with torch.no_grad():
        for data in testloader:
            if count<=100:
                count+=1
            else:
                images, labels = data
                outputs = net(images.cuda())
                _, predicted = torch.max(outputs.data.cpu(), 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
    test_accuracy = 100 * correct / total
    print(test_accuracy)