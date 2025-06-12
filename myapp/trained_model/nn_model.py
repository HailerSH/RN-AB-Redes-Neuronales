
import torch.nn as nn

class NN(nn.Module):
  def __init__(self, input_size=151, hidden_sizes=[512, 256, 128, 64],
                 output_size=1, dropout_rate=0.3):

    super(NN, self).__init__()

    layers =[]
    prev_size = input_size

    for hidden_size in hidden_sizes:
      layers.extend([
        nn.Linear(prev_size, hidden_size),
        nn.BatchNorm1d(hidden_size),
        nn.ReLU(),
        nn.Dropout(dropout_rate)
      ])
      prev_size = hidden_size

    layers.append(nn.Linear(prev_size, output_size))

    self.network = nn.Sequential(*layers)
    self._init_weights()


  def _init_weights(self):
    for module in self.modules():
      if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(module.bias, 0)

  def forward(self, x):
    return self.network(x)