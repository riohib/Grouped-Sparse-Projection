from torch import nn

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        # encoder
        self.e1 = nn.Linear(28 * 28, 128)
        self.e2 = nn.Linear(128, 64)
        self.e3 = nn.Linear(64, 12)
        self.e4 = nn.Linear(12, 3)
        self.relu = nn.ReLU(True)

        # decoder
        self.d1 = nn.Linear(3, 12)
        self.d2 = nn.Linear(12, 64)
        self.d3 = nn.Linear(64, 128)
        self.d4 = nn.Linear(128, 28*28)

        self.tanh = nn.Tanh()

    def forward(self, x):
        # encode
        x = self.relu(self.e1(x))
        x = self.relu(self.e2(x))
        x = self.relu(self.e3(x))
        x = self.relu(self.e4(x))

        # decode
        x = self.relu(self.d1(x))
        x = self.relu(self.d2(x))
        x = self.relu(self.d3(x))
        x = self.tanh(self.d4(x))
        return x