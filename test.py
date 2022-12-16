import torch
import torch.nn as nn

rnn = nn.LSTM(10, 20, 2)

inp = torch.randn(5, 3, 10)
h0 = torch.randn(2, 3, 20)
c0 = torch.randn(2, 3, 20)
output, (hn, cn) = rnn(inp, (h0, c0))

inp = torch.randn(9, 3, 10)
h0 = torch.randn(2, 3, 20)
c0 = torch.randn(2, 3, 20)

output, (hn, cn) = rnn(inp, (h0, c0))