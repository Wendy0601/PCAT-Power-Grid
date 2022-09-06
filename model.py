import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD   
import torch.optim as optim
import numpy as np
from utility import *

class Net(Module):   
    def __init__(self, dim_input  , dim_hidden  , nclass    ):
        super(Net, self).__init__()

        self.cnn_layers = Sequential(
            # Defining a 2D convolution layer
            Conv2d( dim_input , dim_hidden[0] ,  kernel_size=(5,1), stride=1, padding=0),
            BatchNorm2d(dim_hidden[0]), # normalize along the 2-dim if input has four dims 
            ReLU(inplace=True),
            MaxPool2d(kernel_size=(2,1), stride=(2,1)),
            # Defining another 2D convolution layer
            Conv2d(dim_hidden[0], dim_hidden[1], kernel_size=(5,1), stride=1, padding=0),
            BatchNorm2d(dim_hidden[1]),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=(2,1), stride=(2,1)),
            # Defining another 2D convolution layer
            Conv2d(dim_hidden[1], dim_hidden[2], kernel_size= (3,1), stride=1, padding=0),
            BatchNorm2d(dim_hidden[2]),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=(2,1), stride=(2,1)),
            # Defining another 2D convolution layer
            Conv2d(dim_hidden[2], dim_hidden[3], kernel_size= (3,1), stride=1, padding=0),
            BatchNorm2d(dim_hidden[3]),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=(2,1), stride=(2,1)),
        )

        self.linear_layers = Sequential(
            Linear(6 * dim_hidden[3] , nclass)
        )
    
    # Defining the forward pass    
    def forward(self, x):
        x = self.cnn_layers(x) 
        x = x.view(x.size(0), -1) 
        x = self.linear_layers(x)
        return x 
    
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1: 
        torch.nn.init.xavier_uniform_(m.weight) 
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias) 
        
def clamp(X, lower_limit, upper_limit): 
    return torch.max(torch.min(X, upper_limit ), lower_limit)   

def convert_shape(up_limit, batch_size):
    up_limit = torch.FloatTensor(up_limit)
    up_limit = torch.unsqueeze(up_limit, 0)
    up_limit = torch.unsqueeze(up_limit, 0)
    up_limit  = up_limit.repeat(batch_size,1,1,1)
    return up_limit   

def trades_loss( model,
                x_natural,
                y,
                Y_ri,
                optimizer,
                up_limit, 
                down_limit,
                step_size=0.003, 
                perturb_steps=10,
                gamma = 0.02,
                lam=0.1,
                distance='l_inf'):
    # define KL-loss
    criterion_kl = nn.KLDivLoss(size_average=False) 
    Physics_regu = nn.L1Loss(size_average = False)
    model.eval()
    batch_size, _, size_U, _ = x_natural.shape 
    up_limit = convert_shape(up_limit, batch_size)
    down_limit = convert_shape(down_limit, batch_size)
    
    # generate adversarial example 
    delta = torch.randn((batch_size, 1, 2*size_U, 1)).detach()
    delta = torch.min(torch.max(delta,   down_limit),   up_limit) 
    if distance == 'l_inf': 
        x_adv = x_natural +   delta[:, :, :size_U, :]
        for _ in range(perturb_steps): 
            delta.requires_grad_()
            delta_U = delta[:, 0, :size_U, 0]
            delta_I = delta[:, 0, size_U:,0]  
            with torch.enable_grad():
                loss_kl = lam *  criterion_kl(F.log_softmax(model(x_natural \
                         +  delta[:, :, :size_U, :] ), dim=1),F.softmax(model(x_natural), dim=1))\
                - gamma*Physics_regu( (torch.matmul( delta_U , torch.FloatTensor(Y_ri))   ), (delta_I))
            grad = torch.autograd.grad(loss_kl, [delta])[0]
            delta = delta.detach() + step_size * torch.sign(grad.detach()) 
            delta = torch.min(torch.max(delta,   down_limit),   up_limit)
            x_adv = x_natural +  delta[:, :, :size_U, :] 
    else:
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    model.train()  
    x_adv = x_natural.detach()  + delta[:, :, :size_U, :].detach()  
    optimizer.zero_grad() 
    logits = model(x_natural)
    loss_natural = F.cross_entropy(logits, y)
    loss_robust = lam * (1/batch_size) * criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                                    F.softmax(model(x_natural), dim=1)) 
    loss = loss_natural + loss_robust
    return loss , model(x_adv)

def test(test_x, test_labels , model, line_neib ): 
    test_x, test_labels = test_x , test_labels 
    num_test = test_labels.shape[0]
    model.eval()
    output = model(test_x  )  
    criterion = CrossEntropyLoss()  
    loss_test = criterion(output , test_labels ) 
    logit = torch.softmax(output, 1)
    pred = output.max(1)[1] 
    acc_test = torch.eq(pred,test_labels).sum().item()*100/num_test
    multi_labels =  one_hot_neib(test_labels, line_neib) 
    match = 0 
    for i in range(num_test):
        match += multi_labels[i, pred[i]  ]  
    acc_hop = torch.true_divide(match,num_test)
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test ),
         "1-hop accuracy = {:.4f}".format(acc_hop))
    return acc_test, acc_hop
 
def scenario_test(scenario , w, rootPath, model_test):
    if scenario == 1:
        mag = [  '1','1.5','2' , '3' ] 
        name_pre = 'testing_sigPQ_perturb_'  
    elif scenario == 2:
        mag = ['0.1','0.2','0.3','0.4' , '0.5']
        name_pre = 'testing_SigControl_allgener_'  
    else:
        print('Please choose scenario 1 or 2') 
    acc_list = np.zeros((len(mag),1))
    acc_hop_list = np.zeros(acc_list.shape)
    linedata, Y,  line_neib = loadline(rootPath  ) 
    for i in range(len(mag)):
        testName = name_pre + mag[i]  
        print(testName)
        test_x,   test_labels,test_num= load_data_VI_new(w,rootPath, testName)  
        acc, acc_hop = test(test_x, test_labels, model_test, line_neib)
        acc_list[ i] = float("{:.2f}".format(acc)) 
        acc_hop_list[  i] = float("{:.2f}".format(acc_hop))  
    return  acc_list.T 