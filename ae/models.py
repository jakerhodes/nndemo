import torch.nn as nn
import torch.functional as F

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim1 = 800, hidden_dim2 = 400, hidden_dim3 = 100, z_dim = 2):
        super().__init__()

        self.linear  = nn.Linear(input_dim, hidden_dim1)
        self.linear2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.linear3 = nn.Linear(hidden_dim2, hidden_dim3)
        self.mu = nn.Linear(hidden_dim3, z_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        hidden1 = self.relu(self.linear(x))
        hidden2 = self.relu(self.linear2(hidden1))
        hidden3 = self.relu(self.linear3(hidden2))
        z_mu = self.mu(hidden3)
        return z_mu
   
class Decoder(nn.Module):
    def __init__(self, z_dim, output_dim, hidden_dim1 = 100, hidden_dim2 = 400,
                 hidden_dim3 = 800, sigmoid_act = False):
        super().__init__()

        self.linear = nn.Linear(z_dim, hidden_dim1)
        self.linear2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.linear3 = nn.Linear(hidden_dim2, hidden_dim3)
        self.out = nn.Linear(hidden_dim3, output_dim)
        self.relu = nn.ReLU()
        self.sigmoid_act = sigmoid_act

    def forward(self, x):
        hidden1 = self.relu(self.linear(x))
        hidden2 = self.relu(self.linear2(hidden1))
        hidden3 = self.relu(self.linear3(hidden2))
        if self.sigmoid_act == False:
            predicted = (self.out(hidden3))
        else:
            predicted = F.sigmoid(self.out(hidden3))
        return predicted

class AutoEncoder(nn.Module):
    def __init__(self, Encoder, Decoder):
        super().__init__()
        self.Encoder = Encoder
        self.Decoder = Decoder

    def forward(self, x):
        embedding      = self.Encoder(x)
        reconstruction = self.Decoder(embedding)
        return reconstruction    



class ClassificationHead(nn.Module):
    def __init__(self, Encoder, input_dim = 2, hidden_dim1 = 100, hidden_dim2 = 50, output_dim = 10):
        super().__init__()

        self.Encoder = Encoder
        self.linear  = nn.Linear(input_dim, hidden_dim1)
        self.linear2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.out     = nn.Linear(hidden_dim2, output_dim)
        self.relu    = nn.ReLU()

    
    def forward(self, x):
        x = self.relu(self.Encoder(x))
        x = self.relu(self.linear(x))
        x = self.relu(self.linear2(x))
        x = self.out(x)

        return x

class SupervisedAE(nn.Module):
    def __init__(self, Encoder, Decoder, Classifier):
        super().__init__()

        self.Encoder = Encoder
        self.Decoder = Decoder
        self.Classifier = Classifier

    def forward(self, x):

        enc = self.Encoder(x)
        recon = self.Decoder(enc)
        preds = self.Classifier(x)

        return recon, preds


