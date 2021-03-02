
import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, base_channels):
        """
        This is the ResBlock section
        :param base_channels: number_of_features
        """
        super().__init__()

        self.conv1 = nn.Conv2d(base_channels, base_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(base_channels)
        self.lrelu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(base_channels, base_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(base_channels)

    def forward(self, input_tensor):
        identity = self.conv1(input_tensor)
        identity = self.bn1(identity)
        identity = self.lrelu(identity)
        identity = self.conv2(identity)
        identity = self.bn2(identity)
        identity = identity + input_tensor
        identity = self.lrelu(identity)
        return identity


class Conv(nn.Module):
    """This class is the convolution section. It contains three parts convolution,
        batchnorm and activation
    """
    def __init__(self, in_channels, out_channels, dropout=False, activation="relu",
                 dropout_factor=0.1, kernel_size=3, stride=1, padding=1):
        """

        General Formula: output_size = [((input_size + 2*padding - kernel)//stride) + 1)]
        [] represents absolute integer

        :param in_channels: The input channels for convolution
        :param out_channels: The output channels for convolution
        :param kernel_size: The kernel size for height and width, default: 3
        :param stride: The stride value for convolution
        :param padding: The amount of padding required for convolution
        """
        super().__init__()
        conv_modules = nn.ModuleList()

        conv_modules.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))
        conv_modules.append(nn.BatchNorm2d(out_channels))
        if activation == "tanh":
            conv_modules.append(nn.Tanh())
        elif activation == "sigmoid":
            conv_modules.append(nn.Sigmoid())
        if dropout:
            conv_modules.append(nn.Dropout(dropout_factor, inplace=True))
        if activation == "relu":
            conv_modules.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*conv_modules)

    def forward(self, input_tensor):
        """
        The forward function of tensor
        :param input_tensor: This is the input tensor
        :return: input_tensor
        """
        input_tensor = self.conv(input_tensor)
        return input_tensor

class VGG(nn.Module):
    """This is vgg class """

    def __init__(self, in_channels, out_channels, base_channels,
                 n_conv_sections, input_shape):
        """

        :param in_channels: The input channels for vgg network
        :param out_channels: The output channels of vgg network
        :param base_channels: The number of features
        :param n_conv_sections: The number of convolution sections
        :param input_shape: The shape of the input image

        base_channels = 16, input_channels=1, output_channels=1, image_shape = 16*16
        """
        super().__init__()
        # [batch_size, 1, 16, 16]
        self.conv_first = nn.Sequential(Conv(in_channels=in_channels, out_channels=base_channels))
        # [batch_size, 16, 16, 16]

        conv_base = nn.ModuleList()
        for _ in range(n_conv_sections):
            conv_base.append(Conv(in_channels=base_channels, out_channels=base_channels*2))
            conv_base.append(ResBlock(base_channels=base_channels*2))
            conv_base.append(nn.MaxPool2d(2, 2))
            base_channels = base_channels * 2
            input_shape = input_shape // 2

        self.conv_n = nn.Sequential(*conv_base)
        self.conv_final = nn.Sequential(Conv(in_channels=base_channels, out_channels=base_channels,
                                             activation="tanh"))

        linear_list = nn.ModuleList()
        linear_list.append(nn.Linear(base_channels*input_shape*input_shape, 4096))
        linear_list.append(nn.Linear(4096, 1000))
        linear_list.append(nn.Linear(1000, out_channels))
        self.linear = nn.Sequential(*linear_list)

        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_tensor):
        input_tensor = self.conv_first(input_tensor)
        input_tensor = self.conv_n(input_tensor)
        input_tensor = self.conv_final(input_tensor)
        input_tensor = input_tensor.view(input_tensor.size(0), -1)
        input_tensor = self.softmax(self.linear(input_tensor))
        return input_tensor



