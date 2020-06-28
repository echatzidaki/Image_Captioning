import torch
import torch.nn as nn
import torchvision.models as models


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

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        #resnet = models.resnet152(pretrained=True)
        resnet = models.resnet50(pretrained=True)
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
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
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
        
        # Initialize weights
        #torch.nn.init.xavier_uniform(self.linear.weight)
        self.linear.weight.data.uniform_(0.0, 1.0).cuda()
        self.linear.bias.data.fill_(0).cuda()
    
    def w_b_init(self):
        "Initialize weights: xavier and bias: zero"
        # if isinstance(layer, nn.Conv2d):
        #     torch.nn.init.xavier_uniform_(layer.weight).cuda()
        #     self.layer.bias.data.fill_(0).cuda()
        torch.nn.init.xavier_uniform_(self.linear.weight).cuda()
        self.linear.bias.data.fill_(0).cuda()
            
    def hidden_init(self, n_batch):
        "Initialize hidden  and cell states"
#         return (torch.zeros(self.n_layers, n_batch, self.hidden_dim), 
#                torch.zeros(self.n_layers, batch_size, self.hidden_dim))
        return (torch.zeros(self.n_layers, n_batch, self.hidden_dim).cuda(), 
               torch.zeros(self.n_layers, batch_size, self.hidden_dim).cuda())

    
    def forward(self, features, captions):
        
        
        batch_size = features.size(0)
        # initialize hidden state (h, c)
        # h = hidden_init(1)
        
        captions = captions[:,:-1] 
        captions = self.embed(captions)
        
        inputs = torch.cat((features.unsqueeze(1), captions), 1)    
        lay_lstm, hidden = self.lstm(inputs)
        self.hidden = hidden
        
        outputs = self.linear(lay_lstm)
                 
        return outputs


    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        
        if cuda:
            self.cuda()
        else:
            self.cpu()
                                    
        # The output
        sentence = []
        
        # 1 dim: the batch_size
        
        # initialize hidden state (h, c)
        h = hidden_init(1)
        
        for _ in range(max_len):
            x, (h, c) = self.lstm(inputs, hc)
            x = self.linear(x.squeeze(1))
            x_max = x.max(1)[1]
            x_item = x_max.cpu().numpy()[0]
            sentence.append(x_item.item())
            #end token or max length
            if (max_out == 1 or len(x) >= max_len):
                break
                
        inputs = self.embed(x_max).unsqueeze(1)
        
        return sentence 