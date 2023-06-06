import torch.nn as nn
import torch.nn.functional as F


def fc_block(in_size, out_size, dropout,*args, **kwargs):
    return nn.Sequential(
        nn.Linear(in_size, out_size, *args, **kwargs),
        nn.ReLU(),
        nn.Dropout(dropout),
    )

class MLPv1(nn.Module):
    def __init__(self, input_size=3, hidden_size=512, output_size=1, dropout=0, num_layers=5):
        super(MLPv1, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout = dropout
        self.num_layers = num_layers
        fc_blocks = [fc_block(self.hidden_size,self.hidden_size, self.dropout) for i in range(self.num_layers)]
        self.fc_in = nn.Linear(self.input_size, self.hidden_size)
        self.fc_out = nn.Linear(self.hidden_size, self.output_size)
        self.fc = nn.Sequential(*fc_blocks)
        self.dropout_layer = nn.Dropout(self.dropout)

    def forward(self, x):
        # flatten image
        x = x.view(-1, self.input_size)
        x = F.relu(self.fc_in(x))
        x = self.dropout_layer(x)
        x = self.fc(x)
        # add output layer
        x = self.fc_out(x)
        return x
    
