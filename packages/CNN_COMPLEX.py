import torch
from torch import nn

import complexPyTorch
from complexPyTorch.complexLayers import ComplexReLU, ComplexBatchNorm2d, ComplexMaxPool2d, ComplexConv2d, ComplexLinear, ComplexDropout2d


class CNN_COMPLEX(nn.Module):
    def __init__(self, input_dim, n_classes=2, device=torch.device('cpu')):
        super(CNN_COMPLEX, self).__init__()
        self.conv1 = nn.Sequential(
            ComplexConv2d(input_dim, 16, 3),
            ComplexReLU(),
            ComplexMaxPool2d(2),
            ComplexBatchNorm2d(16)
        )
        self.conv2 = nn.Sequential(
            ComplexConv2d(16, 32, 3),
            ComplexReLU(),
            ComplexMaxPool2d(2),
            ComplexBatchNorm2d(32)
        )
        self.conv3 = nn.Sequential(
            ComplexConv2d(32, 64, 3),
            ComplexReLU(),
            ComplexMaxPool2d(2),
            ComplexBatchNorm2d(64)
        )
        self.conv4 = nn.Sequential(
            ComplexConv2d(64, 32, 3),
            ComplexReLU(),
            ComplexMaxPool2d(2),
            ComplexBatchNorm2d(32)
        )
        self.dropout = ComplexDropout2d(0.5, device)
        self.flatten = nn.Flatten().to(torch.cfloat)
        self.fc = nn.Sequential(
            ComplexLinear(192, 64),
            ComplexLinear(64, 32),
            ComplexLinear(32, n_classes)
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.dropout(out)
        out = self.flatten(out)
        out = self.fc(out)
        out = out.abs()
        return out


class CNN_COMPLEX_old(nn.Module):
    def __init__(self, input_dim):
        super(CNN_COMPLEX_old, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_dim, 16, 3).to(torch.cfloat),
            ComplexReLU(),
            ComplexMaxPool2d(2),
            ComplexBatchNorm2d(16)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 3).to(torch.cfloat),
            ComplexReLU(),
            ComplexMaxPool2d(2),
            ComplexBatchNorm2d(32)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 3).to(torch.cfloat),
            ComplexReLU(),
            ComplexMaxPool2d(2),
            ComplexBatchNorm2d(64)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 32, 3).to(torch.cfloat),
            ComplexReLU(),
            ComplexMaxPool2d(2),
            ComplexBatchNorm2d(32)
        )
        self.flatten = nn.Flatten().to(torch.cfloat)
        self.fc = nn.Sequential(
            nn.Linear(192, 64).to(torch.cfloat),
            nn.Linear(64, 32).to(torch.cfloat),
            nn.Linear(32, 16).to(torch.cfloat)
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.flatten(out)
        out = self.fc(out)
        out = out.abs()
        return out


class CNN_REAL(nn.Module):
    def __init__(self, input_dim, n_classes=2):
        super(CNN_REAL, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_dim, 16, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(16)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(32)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(64)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(32)
        )
        self.dropout = nn.Dropout2d(0.5)
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(192, 64),
            nn.Linear(64, 32),
            nn.Linear(32, n_classes)
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.dropout(out)
        out = self.flatten(out)
        out = self.fc(out)
        return out


class CNN_HYBRID(nn.Module):
    def __init__(self, device, complex_state_dict=None, real_state_dict=None):
        super(CNN_HYBRID, self).__init__()

        self.cnn_complex = CNN_COMPLEX(1, 16, device=device)
        self.cnn_real = CNN_REAL(1, 16)

        if complex_state_dict:
            # the loaded state_dict ends with 2 classes, so we replace the last layer parameters with random values
            complex_state_dict.pop('fc.2.fc_i.bias')
            complex_state_dict['fc.2.fc_i.bias'] = torch.rand([16])
            complex_state_dict.pop('fc.2.fc_r.bias')
            complex_state_dict['fc.2.fc_r.bias'] = torch.rand([16])
            complex_state_dict.pop('fc.2.fc_i.weight')
            complex_state_dict['fc.2.fc_i.weight'] = torch.rand([16, 32])
            complex_state_dict.pop('fc.2.fc_r.weight')
            complex_state_dict['fc.2.fc_r.weight'] = torch.rand([16, 32])

            self.cnn_complex.load_state_dict(complex_state_dict)
        if real_state_dict:
            real_state_dict.pop('fc.2.bias')
            real_state_dict['fc.2.bias'] = torch.rand([16])
            real_state_dict.pop('fc.2.weight')
            real_state_dict['fc.2.weight'] = torch.rand([16, 32])

            self.cnn_real.load_state_dict(real_state_dict)

        self.fc = nn.Sequential(
            nn.Linear(32, 16),
            nn.Linear(16, 2)
        )

    def forward(self, x):
        c_tensor = x[:, 0].unsqueeze(dim=1)
        out_c = self.cnn_complex(c_tensor)
        r_tensor = x[:, 1].unsqueeze(dim=1).real
        out_r = self.cnn_real(r_tensor)
        cat = torch.cat([out_c, out_r], dim=1)
        out = self.fc(cat)
        return out


# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# complex_state_dict = torch.load(
#     f='models/CYCLES_CNN(Complex + Mel scale + No Normalization)/CYCLES_CNN(Complex + Mel scale + No Normalization)_final.pth')
# real_state_dict = torch.load(
#     f='models/CYCLES_CNN(Real + Mel scale + Not Normalization)/CYCLES_CNN(Real + Mel scale + Not Normalization)_final.pth')

# model = CNN_HYBRID(device=device, complex_state_dict=complex_state_dict,
#                    real_state_dict=real_state_dict).to(device)

# c_tensor = torch.rand([128, 1, 128, 51]).to(torch.cfloat).to(device)
# r_tensor = torch.rand([128, 1, 128, 51]).to(device)

# tensor = torch.cat([c_tensor, r_tensor], dim=1).to(device)

# output = model(tensor)
# print(output.shape)
