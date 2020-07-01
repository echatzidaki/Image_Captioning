import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable


# https://pytorch.org/docs/master/torchvision/models.html
# resnet18 = models.resnet18(pretrained=True)
# alexnet = models.alexnet(pretrained=True)
# squeezenet = models.squeezenet1_0(pretrained=True)
# vgg16 = models.vgg16(pretrained=True)
# densenet = models.densenet161(pretrained=True)
# inception = models.inception_v3(pretrained=True)
# googlenet = models.googlenet(pretrained=True)
# shufflenet = models.shufflenet_v2_x1_0(pretrained=True)
# mobilenet = models.mobilenet_v2(pretrained=True)
# resnext50_32x4d = models.resnext50_32x4d(pretrained=True)
# wide_resnet50_2 = models.wide_resnet50_2(pretrained=True)
# mnasnet = models.mnasnet1_0(pretrained=True)
# torchvision.models.resnet152  https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/reports/custom/15721427.pdf

#TODO: beam search, attention

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        # resnet = models.resnet152(pretrained=True)
        resnet = models.resnet18(pretrained=True)
        # resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1, drop_prob=0.5):
        super().__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.drop_prob = drop_prob
        
        
        # Define Embedding
        self.embed = nn.Embedding(vocab_size, embed_size)
        
        # Define LSTM - state condition - UserWarning
        if num_layers > 1:
            self.lstm = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, 
                            num_layers=num_layers, dropout=drop_prob, batch_first=True)
        else:
            self.lstm = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, 
                            num_layers=num_layers, dropout=0, batch_first=True)
        
        # Define the Fully-Connected layer
        self.linear = nn.Linear(hidden_size, vocab_size)
        
        # Apply logSoftmax 
        self.softmax = nn.LogSoftmax(1)
        
        self.w_b_init()


    def w_b_init(init_type='kaiming-normal'):
        # https://pytorch.org/docs/stable/nn.init.html
        def init_func(m):
            classname = m.__class__.__name__
            if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
                if init_type == 'normal':
                    init.normal(m.weight.data, 0.0, 0.02).cuda()
                elif init_type == 'uniform':
                    init.uniform_(m.weight.data, a=0.0, b=1.0).cuda()
                elif init_type == 'xavier-normal':
                    init.xavier_normal(m.weight.data, gain=math.sqrt(2)).cuda()
                elif init_type == 'xavier-uniform':
                    init.xavier_uniform_(m.weight.data, gain=1.0).cuda()
                elif init_type == 'kaiming-normal':
                    init.kaiming_normal(m.weight.data, a=0, mode='fan_in').cuda()
                elif init_type == 'kaiming-uniform':
                    init.kaiming_uniform_(m.weight.data, a=0, mode='fan_in').cuda()
                elif init_type == 'orthogonal':
                    init.orthogonal(m.weight.data, gain=math.sqrt(2)).cuda()  
                elif init_type == 'default':
                    pass
                else:
                    assert 0, "Unsupported initialization: {}".format(init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant(m.bias.data, 0.0)
            elif (classname.find('Norm') == 0):
                if hasattr(m, 'weight') and m.weight is not None:
                    init.constant(m.weight.data, 1.0)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant(m.bias.data, 0.0)
        return init_func 

            
    def hidden_init(self, n_batch):
        "Initialize hidden  and cell states"
    #     return (torch.zeros(self.n_layers, n_batch, self.hidden_dim), 
    #            torch.zeros(self.n_layers, batch_size, self.hidden_dim))
        states = (
            Variable(torch.zeros(self.num_layers, n_batch, self.hidden_size), requires_grad=False).cuda(),
            Variable(torch.zeros(self.num_layers, n_batch, self.hidden_size), requires_grad=False).cuda()
        )
#         return (torch.zeros(self.num_layers, n_batch, self.hidden_dim).cuda(), 
#                torch.zeros(self.num_layers, batch_size, self.hidden_dim).cuda())
        return states

    
    def forward(self, features, captions):        
        
        batch_size = features.size(0)
        
        captions = captions[:,:-1] 
        captions = self.embed(captions)
        
        self.hidden = self.hidden_init(batch_size) 
        
        inputs = torch.cat((features.unsqueeze(1), captions), 1)    
        lay_lstm, hidden = self.lstm(inputs)
        
        outputs = self.linear(lay_lstm)
        outputs = self.softmax(outputs)
                 
        return outputs


    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
         
        self.cuda()
        batch_size = inputs.shape[0]
        hidden = self.hidden_init(batch_size)
        # The output
        sentence = []        
        for ml in range(max_len):
            # lstm
            x, hidden = self.lstm(inputs, hidden)
            # linear
            x = self.linear(x).squeeze(1)
            _, x_max = torch.max(x, dim=1)
            x_item = x_max.cpu().numpy()[0].item()
            sentence.append(x_item)
            #end token or max length
            if (x_max == 1 or len(sentence) >= max_len):
                break
                
            inputs = self.embed(x_max)        
            inputs = inputs.unsqueeze(1)
        return sentence 

           

