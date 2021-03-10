import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair

class ResBlock_Conv(nn.Module):
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
            conv_modules.append(nn.PReLU())
        self.conv = nn.Sequential(*conv_modules)

    def forward(self, input_tensor):
        """
        The forward function of tensor
        :param input_tensor: This is the input tensor
        :return: input_tensor
        """
        input_tensor = self.conv(input_tensor)
        return input_tensor


class VGG_Conv(nn.Module):
    """"""
    def __init__(self, in_channels, base_channels,
                 n_conv_sections, dropout=False, dropout_factor=0.1):
        super().__init__()
        self.conv_first = nn.Sequential(Conv(in_channels=in_channels, out_channels=base_channels,
                                             dropout=dropout, dropout_factor=dropout_factor))

        # [batch_size, 16, 16, 16]
        conv_base = nn.ModuleList()
        for _ in range(n_conv_sections):
            conv_base.append(Conv(in_channels=base_channels, out_channels=base_channels * 2, dropout=dropout,
                                  dropout_factor=dropout_factor))
            conv_base.append(ResBlock_Conv(base_channels=base_channels * 2))
            conv_base.append(nn.MaxPool2d(2, 2))
            base_channels = base_channels * 2

        self.conv_n = nn.Sequential(*conv_base)
        self.conv_final = nn.Sequential(Conv(in_channels=base_channels, out_channels=base_channels,
                                             activation="tanh", dropout=dropout, dropout_factor=dropout_factor))
    def forward(self, input_tensor):
        """

        :param input_tensor:
        :return:
        """
        input_tensor = self.conv_first(input_tensor)
        input_tensor = self.conv_n(input_tensor)
        input_tensor = self.conv_final(input_tensor)
        return input_tensor

class VGG(nn.Module):
    """This is vgg class """

    def __init__(self, in_channels, out_channels, base_channels,
                 n_conv_sections, input_shape):
        """

        :param in_channels: The input channels for vgg network
        :param out_channels: The output channels of vgg network
        :param base_channels: The number of features
        :param n_conv_sections: The number of convolution sections. max 4
        :param input_shape: The shape of the input image

        base_channels = 16, input_channels=1, output_channels=1, image_shape = 16*16
        """
        super().__init__()
        # [batch_size, 1, 16, 16]
        n_conv_sections = min(n_conv_sections, 3)
        self.vgg_conv = nn.Sequential(VGG_Conv(in_channels=in_channels, base_channels=base_channels,
                                               n_conv_sections=n_conv_sections, input_shape=input_shape))

        base_channels = base_channels // 2**n_conv_sections
        input_shape = input_shape // 2**n_conv_sections
        linear_list = nn.ModuleList()
        linear_list.append(nn.Linear(base_channels * input_shape * input_shape, 1024))
        linear_list.append(nn.PReLU())
        linear_list.append(nn.Linear(1024, 512))
        linear_list.append(nn.PReLU())
        linear_list.append(nn.Linear(512, out_channels))
        self.linear = nn.Sequential(*linear_list)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_tensor):
        input_tensor = self.vgg_conv(input_tensor)
        input_tensor = input_tensor.view(input_tensor.size(0), -1)
        input_tensor = self.softmax(self.linear(input_tensor))
        return input_tensor

class Inception_Conv(nn.Module):
    """This class creates an inception of 3 conv. inspiration from inception network"""
    def __init__(self, base_channels, dropout, dropout_factor):
        """

        :param base_channels:
        :param dropout:
        :param dropout_factor:
        """
        super().__init__()
        self.conv1_1 = nn.Sequential(Conv(base_channels, base_channels, dropout=dropout, kernel_size=1, stride=1, padding=0,
                            dropout_factor=dropout_factor))
        self.conv3_3 = nn.Sequential(Conv(base_channels, base_channels, dropout=dropout, kernel_size=3, stride=1, padding=1,
                            dropout_factor=dropout_factor))
        self.conv5_5 = nn.Sequential(Conv(base_channels, base_channels, dropout=dropout, kernel_size=5, stride=1, padding=2,
                            dropout_factor=dropout_factor))
        self.conv7_7 = nn.Sequential(Conv(base_channels, base_channels, dropout=dropout, kernel_size=7, stride=1, padding=3,
                            dropout_factor=dropout_factor))
        self.out_conv = nn.Sequential(Conv(base_channels*4, base_channels*2, dropout=dropout, kernel_size=1,
                                           stride=1, padding=0, dropout_factor=dropout_factor))
    def forward(self, input_tensor):
        """

        :param input_tensor:
        :return:
        """
        out_1_1 = self.conv1_1(input_tensor)
        out_3_3 = self.conv3_3(input_tensor)
        out_5_5 = self.conv5_5(input_tensor)
        out_7_7 = self.conv7_7(input_tensor)

        final_output_tensor = torch.cat([out_1_1, out_3_3, out_5_5, out_7_7], dim=1)
        final_output_tensor = self.out_conv(final_output_tensor)
        return final_output_tensor

class Inception_Model(nn.Module):
    """"""
    def __init__(self, in_channels, base_channels, n_layers, dropout, dropout_factor):
        """

        :param in_channels:
        :param out_channels:
        :param base_channels:
        :param n_layers: max upto 4
        """
        super().__init__()
        n_layers = min(n_layers, 3)
        self.conv_first = nn.Sequential(Conv(in_channels, base_channels, dropout=dropout, dropout_factor=dropout_factor,
                               kernel_size=3, stride=1, padding=1))
        conv_n = nn.ModuleList()
        for _ in range(n_layers):
            conv_n.append(Inception_Conv(base_channels,
                                dropout=dropout, dropout_factor=dropout_factor))
            conv_n.append(nn.MaxPool2d(2, 2))
            base_channels = base_channels*2

        self.conv_n = nn.Sequential(*conv_n)

        self.conv_final = nn.Sequential(Conv(base_channels, base_channels, dropout=dropout,
                                             dropout_factor=dropout_factor, kernel_size=3, stride=1, padding=1))

    def forward(self, input_tensor):
        """

        :param input_tensor:
        :return:
        """
        input_tensor = self.conv_first(input_tensor)
        input_tensor = self.conv_n(input_tensor)
        input_tensor = self.conv_final(input_tensor)
        return input_tensor


class ResBlock(nn.Module):
    """
    Residual Convolution Block
    """

    def __init__(self, nf):
        """

        :param nf: no of input features
        """
        super().__init__()
        self.conv1 = nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(
            nf + nf, nf, kernel_size=3, stride=1, padding=1, bias=True
        )
        self.conv3 = nn.Conv2d(
            nf + 2 * nf, nf, kernel_size=3, stride=1, padding=1, bias=True
        )
        self.conv4 = nn.Conv2d(
            nf + 3 * nf, nf, kernel_size=3, stride=1, padding=1, bias=True
        )
        self.conv5 = nn.Conv2d(
            nf + 4 * nf, nf, kernel_size=3, stride=1, padding=1, bias=True
        )
        self.p_relu = nn.PReLU()

    def forward(self, input_tensor):
        """

        :param input_tensor: input tensor
        :return:  returns a tensor which is of same shape with added features
        """
        x_1 = self.p_relu(self.conv1(input_tensor))
        x_2 = self.p_relu(self.conv2(torch.cat((input_tensor, x_1), 1)))
        x_3 = self.p_relu(self.conv3(torch.cat((input_tensor, x_1, x_2), 1)))
        x_4 = self.p_relu(self.conv4(torch.cat((input_tensor, x_1, x_2, x_3), 1)))
        x_5 = self.p_relu(self.conv5(torch.cat((input_tensor, x_1, x_2, x_3, x_4), 1)))
        return x_5 * 0.2 + input_tensor


class RRDB(nn.Module):
    """
    Creates the RRDB module from the residual blocks
    """

    def __init__(self, in_channels, base_channels, n_layers, dropout, dropout_factor):
        """

        :param base_channels: No of input features
        """
        super().__init__()
        n_layers = min(n_layers, 3)
        self.conv_first = nn.Sequential(Conv(in_channels, base_channels, dropout=dropout, dropout_factor=dropout_factor,
                                             kernel_size=3, stride=1, padding=1))
        rrdb = nn.ModuleList()
        for _ in range(n_layers):
            rrdb.append(ResBlock(base_channels))

        conv_n = nn.ModuleList()
        for _ in range(n_layers):
            conv_n.append(Conv(base_channels, base_channels*2, kernel_size=1, stride=1, padding=0))
            conv_n.append(nn.MaxPool2d(2, 2))
            base_channels = base_channels*2
        self.conv_n = nn.Sequential(*conv_n)

        self.rrdb = nn.Sequential(*rrdb)

        self.conv_final = nn.Sequential(Conv(base_channels, base_channels, dropout=dropout,
                                             dropout_factor=dropout_factor, kernel_size=3, stride=1, padding=1))


    def forward(self, input_tensor):
        """

        :param  input_tensor: input tensor
        :return:  returns a tensor which is of same shape with
                  added features after Three RRDB passes
        """
        input_tensor = self.conv_first(input_tensor)
        out = self.rrdb(input_tensor)
        input_tensor = out * 0.2 + input_tensor
        input_tensor = self.conv_n(input_tensor)
        input_tensor = self.conv_final(input_tensor)
        return input_tensor



class Fully_Connected(nn.Module):
    """This is a fully connected layer"""

    def __init__(self, in_features, out_features, base_features,
                 n_layers):
        """

        :param in_channels:
        :param out_channels:
        :param base_channels:
        :param n_conv_sections:
        :param input_shape:
        """
        super().__init__()
        self.linear_first = nn.Linear(in_features, base_features)

        linear_n = nn.ModuleList()
        for _ in range(n_layers):
            linear_n.append(nn.Linear(in_features, base_features))
            linear_n.append(nn.PReLU)

        self.linear_n = nn.Sequential(*linear_n)

        self.linear_last = nn.Linear(base_features, out_features)
        self.final_act = nn.LogSoftmax()
        self.act = nn.PReLU()

    def forward(self, input_tensor):
        """

        :param input_tensor:
        :return:
        """
        input_tensor = self.act(self.linear_first(input_tensor))
        input_tensor = self.linear_n(input_tensor)
        input_tensor = self.final_act(self.linear_last(input_tensor))
        return input_tensor



class LocallyConnected2d(nn.Module):
    def __init__(self, in_channels, out_channels, output_size, kernel_size, stride, bias=False):
        """

        :param in_channels:
        :param out_channels:
        :param output_size:
        :param kernel_size:
        :param stride:
        :param bias:
        """
        super().__init__()
        output_size = _pair(output_size)
        self.weight = nn.Parameter(
            torch.randn(1, out_channels, in_channels, output_size[0], output_size[1], kernel_size ** 2)
        )
        if bias:
            self.bias = nn.Parameter(
                torch.randn(1, out_channels, output_size[0], output_size[1])
            )
        else:
            self.register_parameter('bias', None)
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)

    def forward(self, x):
        _, c, h, w = x.size()
        kh, kw = self.kernel_size
        dh, dw = self.stride
        x = x.unfold(2, kh, dh).unfold(3, kw, dw)
        x = x.contiguous().view(*x.size()[:-2], -1)
        # Sum in in_channel and kernel_size dims
        out = (x.unsqueeze(1) * self.weight).sum([2, -1])
        if self.bias is not None:
            out += self.bias
        return out

class Locally_Connected_Module(nn.Module):
    """This architecture is for locally connected without share weight"""
    def __init__(self, in_channels, base_channels, n_conv_layers, dropout=False, dropout_factor=0.1, input_shape=16):
        """

        :param in_channels:
        :param base_channels:
        :param out_channels:
        :param n_conv_layers:
        """
        super().__init__()
        self.local_conv1 = nn.Sequential(LocallyConnected2d(in_channels, base_channels, input_shape-2, 3, 1, True))
        self.local_conv2 = nn.Sequential(LocallyConnected2d(base_channels, base_channels, input_shape-4, 3, 1, True))
        self.local_conv3 = nn.Sequential(LocallyConnected2d(base_channels, base_channels, input_shape-6, 3, 1, True))

        self.act = nn.PReLU()

        n_conv_layers = min(n_conv_layers, 3)
        layer_n = nn.ModuleList()
        for _ in range(n_conv_layers):
            layer_n.append(Conv(base_channels, base_channels*2, dropout=dropout, dropout_factor=dropout_factor))
            layer_n.append(nn.MaxPool2d(2, 2))
            base_channels = base_channels * 2

        self.conv_layer_n = nn.Sequential(*layer_n)
        self.conv_final = nn.Sequential(Conv(base_channels, base_channels, dropout=dropout, dropout_factor=dropout_factor,
                                             activation="tanh"))

    def forward(self, input_tensor):
        """

        :param input_tensor:
        :return:
        """
        input_tensor = self.act(self.local_conv1(input_tensor))
        input_tensor = self.act(self.local_conv2(input_tensor))
        input_tensor = self.act(self.local_conv3(input_tensor))

        input_tensor = self.conv_layer_n(input_tensor)

        input_tensor = self.conv_final(input_tensor)
        return input_tensor

class Locally_Connected_Network(nn.Module):
    """"""
    def __init__(self, in_channels, out_channels, base_channels,
                 n_conv_sections, input_shape=16, dropout=False, dropout_factor=0.1):
        """

        :param in_channels:
        :param out_channels:
        :param base_channels:
        :param n_conv_sections:
        :param input_shape:
        :param dropout:
        :param dropout_factor:
        """
        super().__init__()
        self.local_connected_module = Locally_Connected_Module(in_channels=in_channels, base_channels=base_channels,
                                                               n_conv_layers=n_conv_sections,
                                                               dropout=dropout, dropout_factor=dropout_factor,
                                                               input_shape=input_shape)
        base_channels = base_channels * 2 ** n_conv_sections
        input_shape = (input_shape-6) // 2 ** n_conv_sections
        linear_list = nn.ModuleList()
        linear_list.append(nn.Linear(base_channels * input_shape * input_shape, 1024))
        linear_list.append(nn.PReLU())
        linear_list.append(nn.Linear(1024, 512))
        linear_list.append(nn.PReLU())
        linear_list.append(nn.Linear(512, out_channels))
        self.linear = nn.Sequential(*linear_list)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_tensor):
        """

        :param input_tensor:
        :return:
        """
        input_tensor = self.local_connected_module(input_tensor)
        input_tensor = input_tensor.view(input_tensor.size(0), -1)
        input_tensor = self.softmax(self.linear(input_tensor))
        return input_tensor



class Inception_Res_Module(nn.Module):
    """"""
    def __init__(self, base_channels):
        """

        :param in_channels:
        :param base_channels:
        :param layers:
        """
        super().__init__()
        self.conv_first = nn.Sequential(Conv(base_channels, base_channels))
        self.model_1 = nn.Sequential(ResBlock_Conv(base_channels))
        self.model_2 = nn.Sequential(ResBlock(base_channels))
        self.final_conv = nn.Sequential(Conv(base_channels*2, base_channels*2))

    def forward(self, input_tensor):
        """

        :param input_tensor:
        :return:
        """
        input_tensor = self.conv_first(input_tensor)
        out_1 = self.model_1(input_tensor)
        out_2 = self.model_2(input_tensor)
        out_3 = torch.cat([out_1, out_2], 1)
        out_3 = self.final_conv(out_3)
        return out_3


class Inception_Res_Network(nn.Module):
    """"""
    def __init__(self, in_channels, base_channels, n_layers, dropout=False, dropout_factor=0.1):
        super().__init__()
        n_layers = min(n_layers, 3)
        self.conv_first = nn.Sequential(Conv(in_channels, base_channels, dropout=dropout, dropout_factor=dropout_factor,
                                             kernel_size=3, stride=1, padding=1))
        conv_n = nn.ModuleList()
        for _ in range(n_layers):
            conv_n.append(Inception_Res_Module(base_channels))
            conv_n.append(nn.MaxPool2d(2, 2))
            base_channels = base_channels * 2

        self.conv_n = nn.Sequential(*conv_n)

        self.conv_final = nn.Sequential(Conv(base_channels, base_channels, dropout=dropout,
                                             dropout_factor=dropout_factor, kernel_size=3, stride=1, padding=1))
    def forward(self, input_tensor):
        input_tensor = self.conv_first(input_tensor)
        input_tensor = self.conv_n(input_tensor)
        input_tensor = self.conv_final(input_tensor)
        return input_tensor


class Ensemble_Network(nn.Module):
    """"""
    def __init__(self, in_channels, base_channels, layers, out_channels, vgg_dropout=False, inception_dropout=False,
                 rrdb_dropout=False,vgg_dropout_factor=0.1, rrdb_dropout_factor=0.1,
                        inception_dropout_factor=0.1, input_shape=16):
        """

        :param in_channels:
        :param base_channels:
        :param layers:
        :param out_channels:
        :param vgg_dropout:
        :param inception_dropout:
        :param rrdb_dropout:
        :param vgg_dropout_factor:
        :param rrdb_dropout_factor:
        :param inception_dropout_factor:
        :param input_shape:
        """

        super().__init__()
        self.vgg_conv = nn.Sequential(VGG_Conv(in_channels, base_channels, layers, vgg_dropout,
                                               vgg_dropout_factor))
        self.inception_conv = nn.Sequential(Inception_Model(in_channels, base_channels, layers,
                                                            inception_dropout, inception_dropout_factor))
        self.rrdb_conv = nn.Sequential(RRDB(in_channels, base_channels, layers, rrdb_dropout, rrdb_dropout_factor))
        self.inception_res = nn.Sequential(Inception_Res_Network(in_channels, base_channels, layers
                                                                 ))

        base_channels = base_channels * 2**min(layers, 3) * 4
        input_shape = input_shape // 2**min(layers, 3)

        linear_list = nn.ModuleList()
        linear_list.append(nn.Linear(base_channels * input_shape * input_shape, 1024))
        linear_list.append(nn.PReLU())
        linear_list.append(nn.Linear(1024, 512))
        linear_list.append(nn.PReLU())
        linear_list.append(nn.Linear(512, out_channels))
        self.linear = nn.Sequential(*linear_list)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_tensor):
        out_1 = self.vgg_conv(input_tensor)
        out_2 = self.inception_conv(input_tensor)
        out_3 = self.rrdb_conv(input_tensor)
        out_4 = self.inception_res(input_tensor)

        out = torch.cat([out_1, out_2, out_3, out_4], 1)
        out = out.view(input_tensor.size(0), -1)
        out = self.softmax(self.linear(out))
        return out












