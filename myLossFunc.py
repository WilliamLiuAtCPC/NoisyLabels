import torch.nn as nn
import torch

class CE(nn.Module):
    # def __init__(self):
    #     print('define myLoss')

    def forward(self,outputs,targets):
        sm = nn.Softmax(dim=1)
        x = sm(outputs)
        # print(x)
        x = torch.log(x)
        size_outputs = outputs.size()
        ce = 0
        for i in range(size_outputs[0]):
            ce += x[i][targets[i]]
        ce /= size_outputs[0]
        return -ce
        # nll = nn.NLLLoss()
        # return nll(x,targets)

class CES_CE(nn.Module):
    def __init__(self,A,a,b):
        super(CES_CE, self).__init__()
        self.A = A
        self.a = a
        self.b = b

    def forward(self,outputs,targets):
        sm = nn.LogSoftmax(dim=1)
        x = sm(outputs)
        y = torch.exp(x)
        ce = 0
        ces = 0
        size_outputs = outputs.size()
        for i in range(size_outputs[0]):
            ce -= x[i][targets[i]]
            ces += y[i][targets[i]]
            for j in range(size_outputs[1]):
                ces -= y[i][j]
        ces=ces*self.A/size_outputs[0]
        ce /= size_outputs[0]
        return (self.a*ce + self.b*ces)

class CES_CE2(nn.Module):
    def __init__(self,A,a,b):
        super(CES_CE2, self).__init__()
        self.A = A
        self.a = a
        self.b = b

    def forward(self,outputs,targets):
        sm = nn.LogSoftmax(dim=1)
        x = sm(outputs)
        y = torch.exp(x)
        # z = self.a*x-self.A*self.b*(y-1)
        # nll = nn.NLLLoss()
        # return nll(z,targets)
        ce = 0
        ces = 0
        size_outputs = outputs.size()
        for i in range(size_outputs[0]):
            ce -= x[i][targets[i]]
            ces += y[i][targets[i]]-1
        ces=ces*self.A/size_outputs[0]
        ce /= size_outputs[0]
        return (self.a*ce + self.b*ces)


class MAE(nn.Module):
    # def __init__(self):
    #     print('define myLoss')

    def forward(self,outputs,targets):
        sm = nn.Softmax(dim=1)
        x = sm(outputs)
        size_outputs = outputs.size()
        mae = 0
        for i in range(size_outputs[0]):
            mae += 2-2*x[i][targets[i]]
        mae /= size_outputs[0]
        return mae