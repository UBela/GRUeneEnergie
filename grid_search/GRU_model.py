import torch
import torch.nn as nn
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class GRUNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, drop_prob=0.2):
        super(GRUNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
    
        self.gru = nn.GRU(
            input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob
        )
        self.fc = nn.Linear(hidden_dim, output_dim)
        #self.relu = nn.ReLU() this does not seem good in regression

    def forward(self, x, h=None):
        if h is None:
          h = self.init_hidden(x.size(0), device)
        out, h = self.gru(x, h)
        out = self.fc(out[:, -1]) 
        return out

    def init_hidden(self, batch_size, device):
        # Initialze h_0 with zeros
        return torch.zeros(self.n_layers, batch_size, self.hidden_dim, device=device)

 