import torch
from torch import nn
import math
from torch.autograd import Function
from torch.nn.utils import weight_norm
import torch.nn.functional as F
from .resnet18 import resnet18
import numpy as np
from S3 import S3

from .utils.Conv_Blocks import Inception_Block_V1
from .utils.Embed import DataEmbedding

from timm.loss import LabelSmoothingCrossEntropy
from timm.models.layers import DropPath
from timm.models.layers import trunc_normal_

# from utils import weights_init

def get_backbone_class(backbone_name):
    """Return the algorithm class with the given name."""
    if backbone_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(backbone_name))
    return globals()[backbone_name]


##################################################
##########  BACKBONE NETWORKS  ###################
##################################################


########## TSLANet #########################

class ICB(nn.Module):
    def __init__(self, in_features, hidden_features, drop=0.):
        super().__init__()
        self.conv1 = nn.Conv1d(in_features, hidden_features, 1)
        self.conv2 = nn.Conv1d(in_features, hidden_features, 3, 1, 1)
        self.conv3 = nn.Conv1d(hidden_features, in_features, 1)
        self.drop = nn.Dropout(drop)
        self.act = nn.GELU()

    def forward(self, x):
        x = x.transpose(1, 2)
        x1 = self.conv1(x)
        x1_1 = self.act(x1)
        x1_2 = self.drop(x1_1)

        x2 = self.conv2(x)
        x2_1 = self.act(x2)
        x2_2 = self.drop(x2_1)

        out1 = x1 * x2_2
        out2 = x2 * x1_2

        x = self.conv3(out1 + out2)
        x = x.transpose(1, 2)
        return x


class PatchEmbed(nn.Module):
    def __init__(self, seq_len, patch_size=8, in_chans=3, embed_dim=384):
        super().__init__()
        stride = patch_size // 2
        num_patches = int((seq_len - patch_size) / stride + 1)
        self.num_patches = num_patches
        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)

    def forward(self, x):
        x_out = self.proj(x).flatten(2).transpose(1, 2)
        return x_out


class Adaptive_Spectral_Block(nn.Module):
    def __init__(self, dim, adaptive_filter=False):
        super().__init__()
        self.complex_weight_high = nn.Parameter(torch.randn(dim, 2, dtype=torch.float32) * 0.02)
        self.complex_weight = nn.Parameter(torch.randn(dim, 2, dtype=torch.float32) * 0.02)

        trunc_normal_(self.complex_weight_high, std=.02)
        trunc_normal_(self.complex_weight, std=.02)
        self.threshold_param = nn.Parameter(torch.rand(1)) # * 0.5)
        self.adaptive_filter = adaptive_filter

    def create_adaptive_high_freq_mask(self, x_fft):
        B, _, _ = x_fft.shape

        # Calculate energy in the frequency domain
        energy = torch.abs(x_fft).pow(2).sum(dim=-1)

        # Flatten energy across H and W dimensions and then compute median
        flat_energy = energy.view(B, -1)  # Flattening H and W into a single dimension
        median_energy = flat_energy.median(dim=1, keepdim=True)[0]  # Compute median
        median_energy = median_energy.view(B, 1)  # Reshape to match the original dimensions

        # Normalize energy
        epsilon = 1e-6  # Small constant to avoid division by zero
        normalized_energy = energy / (median_energy + epsilon)

        adaptive_mask = ((normalized_energy > self.threshold_param).float() - self.threshold_param).detach() + self.threshold_param
        adaptive_mask = adaptive_mask.unsqueeze(-1)

        return adaptive_mask

    def forward(self, x_in):
        B, N, C = x_in.shape

        dtype = x_in.dtype
        x = x_in.to(torch.float32)

        # Apply FFT along the time dimension
        x_fft = torch.fft.rfft(x, dim=1, norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        x_weighted = x_fft * weight

        if self.adaptive_filter:
            # Adaptive High Frequency Mask (no need for dimensional adjustments)
            freq_mask = self.create_adaptive_high_freq_mask(x_fft)
            x_masked = x_fft * freq_mask.to(x.device)

            weight_high = torch.view_as_complex(self.complex_weight_high)
            x_weighted2 = x_masked * weight_high

            x_weighted += x_weighted2

        # Apply Inverse FFT
        x = torch.fft.irfft(x_weighted, n=N, dim=1, norm='ortho')

        x = x.to(dtype)
        x = x.view(B, N, C)  # Reshape back to original shape

        return x


class TSLANet_layer(nn.Module):
    def __init__(self, dim, mlp_ratio=3., drop=0., drop_path=0., norm_layer=nn.LayerNorm, configs=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.asb = Adaptive_Spectral_Block(dim, adaptive_filter=configs.adaptive_filter)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.icb = ICB(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)
        self.configs = configs

    def forward(self, x):
        # Check if both ASB and ICB are true
        if self.configs.ICB and self.configs.ASB:
            x = x + self.drop_path(self.icb(self.norm2(self.asb(self.norm1(x)))))
        # If only ICB is true
        elif self.configs.ICB:
            x = x + self.drop_path(self.icb(self.norm2(x)))
        # If only ASB is true
        elif self.configs.ASB:
            x = x + self.drop_path(self.asb(self.norm1(x)))
        # If neither is true, just pass x through
        return x


class TSLANet(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.masking_ratio = configs.masking_ratio
        self.patch_embed = PatchEmbed(
            seq_len=configs.sequence_len, patch_size=configs.patch_size,
            in_chans=configs.input_channels, embed_dim=configs.emb_dim
        )
        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, configs.emb_dim), requires_grad=True)
        self.pos_drop = nn.Dropout(p=configs.dropout)

        self.input_layer = nn.Linear(configs.patch_size, configs.emb_dim)

        dpr = [x.item() for x in torch.linspace(0, configs.dropout, configs.depth)]  # stochastic depth decay rule

        self.tsla_blocks = nn.ModuleList([
            TSLANet_layer(dim=configs.emb_dim, drop=configs.dropout, drop_path=dpr[i], configs=configs)
            for i in range(configs.depth)]
        )

        # Classifier head
        #self.head = nn.Linear(configs.emb_dim, configs.num_classes)

        # init weights
        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    #From original implementation, Not used in UniDABench
    def pretrain(self, x_in):
        x = self.patch_embed(x_in)
        x = x + self.pos_embed
        x_patched = self.pos_drop(x)

        x_masked, _, self.mask, _ = random_masking_3D(x, mask_ratio=self.masking_ratio)
        self.mask = self.mask.bool()  # mask: [bs x num_patch x n_vars]

        for tsla_blk in self.tsla_blocks:
            x_masked = tsla_blk(x_masked)

        return x_masked, x_patched

    def forward(self, x):
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for tsla_blk in self.tsla_blocks:
            x = tsla_blk(x)

        x = x.mean(1)
        #x = self.head(x)
        return x



########## S3 ##############################

class S3Layer(nn.Module):
    def __init__(self, configs):
        super(S3Layer, self).__init__()
        #self.s3_layers = S3(num_layers=3, initial_num_segments=4, shuffle_vector_dim=1, segment_multiplier=2)
        self.s3_layers = S3(num_layers=8, initial_num_segments=4, shuffle_vector_dim=1, segment_multiplier=2)
        self.cnn = CNN(configs)

    def forward(self, x_in):
        x = self.s3_layers(x_in)
        x_flat = self.cnn(x)
        return x_flat


########## CNN #############################
class CNN(nn.Module):
    def __init__(self, configs):
        super(CNN, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(configs.input_channels, configs.mid_channels, kernel_size=configs.kernel_size,
                      stride=configs.stride, bias=False, padding=(configs.kernel_size // 2)),
            nn.BatchNorm1d(configs.mid_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(configs.dropout)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(configs.mid_channels, configs.mid_channels * 2, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(configs.mid_channels * 2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(configs.mid_channels * 2, configs.final_out_channels, kernel_size=8, stride=1, bias=False,
                      padding=4),
            nn.BatchNorm1d(configs.final_out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )

        self.adaptive_pool = nn.AdaptiveAvgPool1d(configs.features_len)

    def forward(self, x_in):
        x = self.conv_block1(x_in)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.adaptive_pool(x)

        x_flat = x.reshape(x.shape[0], -1)
        return x_flat


########## FNO #############################
class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, fl=128):
        super(SpectralConv1d, self).__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))
        self.pi = torch.acos(torch.zeros(1)).item() * 2

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x = torch.cos(x)
        x_ft = torch.fft.rfft(x, norm='ortho')
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1) // 2 + 1, device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)
        r = out_ft[:, :, :self.modes1].abs()
        p = out_ft[:, :, :self.modes1].angle()
        return torch.concat([r, p], -1), out_ft


class FNO(nn.Module):
    def __init__(self, configs):
        super(FNO, self).__init__()
        self.modes1 = configs.fourier_modes  # Number of low-frequency modes to keep
        self.width = configs.input_channels
        self.length = configs.sequence_len
        self.freq_feature = SpectralConv1d(self.width, self.width, self.modes1,
                                           self.length)  # Frequency Feature Encoder
        self.bn_freq = nn.BatchNorm1d(
            configs.fourier_modes * 2)  # It doubles because frequency features contain both amplitude and phase
        self.cnn = CNN(configs).to('cuda')  # Time Feature Encoder
        self.avg = nn.Conv1d(self.width, 1, kernel_size=3,
                             stride=configs.stride, bias=False, padding=(3 // 2))

    '''def forward(self, x):
        ef, out_ft = self.freq_feature(x)
        print("ef : ", ef.shape)
        avg = self.avg(ef).squeeze()
        print("avg : ", avg.shape)
        ef = F.relu(self.bn_freq(avg))
        et = self.cnn(x)
        f = torch.concat([ef,et],-1)
        return F.normalize(f)'''

    def forward(self, x):
        ef, out_ft = self.freq_feature(x)
        # print(ef.shape)
        # print(self.avg(ef).shape)
        if ef.shape[0] == 1 and ef.shape[1] != 1:
            ef = self.avg(ef).squeeze(0)
        elif ef.shape[1] == 1:
            ef = ef.squeeze()
        else:
            ef = self.bn_freq(self.avg(ef).squeeze())

        ef = F.relu(ef)
        et = self.cnn(x)
        # print(ef.shape, et.shape)
        f = torch.concat([ef, et], -1)
        return F.normalize(f)


class classifier(nn.Module):
    def __init__(self, configs):
        super(classifier, self).__init__()
        self.logits = nn.Linear(configs.features_len * configs.final_out_channels, configs.num_classes)
        self.configs = configs

    def forward(self, x):
        predictions = self.logits(x)

        return predictions


class classifierNoBias(nn.Module):
    def __init__(self, configs):
        super(classifierNoBias, self).__init__()
        self.logits = nn.Linear(configs.features_len * configs.final_out_channels, configs.num_classes, bias=False)
        self.configs = configs

    def forward(self, x):
        predictions = self.logits(x)

        return predictions


class classifierOVANet(nn.Module):
    def __init__(self, configs):
        super(classifierOVANet, self).__init__()
        final_out_channels = configs.final_out_channels
        if configs.isFNO:
            final_out_channels = configs.final_out_channels + 2 * configs.fourier_modes
        self.logits = nn.Linear(configs.features_len * final_out_channels, configs.num_classes * 2)
        self.configs = configs

    def forward(self, x):
        predictions = self.logits(x)

        return predictions


class LinearAverage(nn.Module):
    def __init__(self, inputSize, outputSize, T=0.05, momentum=0.0):
        super(LinearAverage, self).__init__()
        self.nLem = outputSize
        self.momentum = momentum
        self.register_buffer('params', torch.tensor([T, momentum]))
        self.register_buffer('memory', torch.zeros(outputSize, inputSize))
        self.flag = 0
        self.T = T
        self.memory = self.memory.cuda()

    def forward(self, x, y):
        # print(x.shape)
        out = torch.mm(x, self.memory.t()) / self.T
        return out

    def update_weight(self, features, index):
        if not self.flag:
            weight_pos = self.memory.index_select(0, index.data.view(-1)).resize_as_(features)
            weight_pos.mul_(0.0)
            weight_pos.add_(torch.mul(features.data, 1.0))

            w_norm = weight_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_weight = weight_pos.div(w_norm)
            self.memory.index_copy_(0, index, updated_weight)
            self.flag = 1
        else:
            weight_pos = self.memory.index_select(0, index.data.view(-1)).resize_as_(features)
            weight_pos.mul_(self.momentum)
            weight_pos.add_(torch.mul(features.data, 1 - self.momentum))

            w_norm = weight_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_weight = weight_pos.div(w_norm)
            self.memory.index_copy_(0, index, updated_weight)
        self.memory = F.normalize(self.memory)  # .cuda()

    def set_weight(self, features, index):
        self.memory.index_copy_(0, index, features)


########## TCN #############################
torch.backends.cudnn.benchmark = True  # might be required to fasten TCN


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TCN(nn.Module):
    def __init__(self, configs):
        super(TCN, self).__init__()

        in_channels0 = configs.input_channels
        out_channels0 = configs.tcn_layers[1]
        kernel_size = configs.tcn_kernel_size
        stride = 1
        dilation0 = 1
        padding0 = (kernel_size - 1) * dilation0

        self.net0 = nn.Sequential(
            weight_norm(nn.Conv1d(in_channels0, out_channels0, kernel_size, stride=stride, padding=padding0,
                                  dilation=dilation0)),
            nn.ReLU(),
            weight_norm(nn.Conv1d(out_channels0, out_channels0, kernel_size, stride=stride, padding=padding0,
                                  dilation=dilation0)),
            nn.ReLU(),
        )

        self.downsample0 = nn.Conv1d(in_channels0, out_channels0, 1) if in_channels0 != out_channels0 else None
        self.relu = nn.ReLU()

        in_channels1 = configs.tcn_layers[0]
        out_channels1 = configs.tcn_layers[1]
        dilation1 = 2
        padding1 = (kernel_size - 1) * dilation1
        self.net1 = nn.Sequential(
            nn.Conv1d(in_channels0, out_channels1, kernel_size, stride=stride, padding=padding1, dilation=dilation1),
            nn.ReLU(),
            nn.Conv1d(out_channels1, out_channels1, kernel_size, stride=stride, padding=padding1, dilation=dilation1),
            nn.ReLU(),
        )
        self.downsample1 = nn.Conv1d(out_channels1, out_channels1, 1) if in_channels1 != out_channels1 else None

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(in_channels0, out_channels0, kernel_size=kernel_size, stride=stride, bias=False, padding=padding0,
                      dilation=dilation0),
            Chomp1d(padding0),
            nn.BatchNorm1d(out_channels0),
            nn.ReLU(),

            nn.Conv1d(out_channels0, out_channels0, kernel_size=kernel_size, stride=stride, bias=False,
                      padding=padding0, dilation=dilation0),
            Chomp1d(padding0),
            nn.BatchNorm1d(out_channels0),
            nn.ReLU(),
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(out_channels0, out_channels1, kernel_size=kernel_size, stride=stride, bias=False,
                      padding=padding1, dilation=dilation1),
            Chomp1d(padding1),
            nn.BatchNorm1d(out_channels1),
            nn.ReLU(),

            nn.Conv1d(out_channels1, out_channels1, kernel_size=kernel_size, stride=stride, bias=False,
                      padding=padding1, dilation=dilation1),
            Chomp1d(padding1),
            nn.BatchNorm1d(out_channels1),
            nn.ReLU(),
        )

    def forward(self, inputs):
        """Inputs have to have dimension (N, C_in, L_in)"""
        x0 = self.conv_block1(inputs)
        res0 = inputs if self.downsample0 is None else self.downsample0(inputs)
        out_0 = self.relu(x0 + res0)

        x1 = self.conv_block2(out_0)
        res1 = out_0 if self.downsample1 is None else self.downsample1(out_0)
        out_1 = self.relu(x1 + res1)

        out = out_1[:, :, -1]
        return out


######## RESNET ##############################################

class RESNET18(nn.Module):
    def __init__(self, configs):
        super(RESNET18, self).__init__()
        self.resnet = resnet18(configs)

    def forward(self, x_in):
        x = self.resnet(x_in)
        x_flat = x.reshape(x.shape[0], -1)
        return x_flat


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=1, stride=stride,
                               bias=False)
        self.bn1 = nn.BatchNorm1d(planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = F.relu(out)

        return out


##################################################
##########  OTHER NETWORKS  ######################
##################################################

class DiscriminatorUDA(nn.Module):
    """Discriminator model for source domain."""

    def __init__(self, configs):
        """Init discriminator."""
        super(DiscriminatorUDA, self).__init__()
        final_out_channels = configs.final_out_channels
        if configs.isFNO:
            final_out_channels = configs.final_out_channels * 2
            final_out_channels = configs.final_out_channels + 2 * configs.fourier_modes
        self.layer = nn.Sequential(
            nn.Linear(configs.features_len * final_out_channels, configs.disc_hid_dim),
            nn.ReLU(),
            nn.Linear(configs.disc_hid_dim, configs.disc_hid_dim),
            nn.ReLU(),
            nn.Linear(configs.disc_hid_dim, 1),
            nn.Sigmoid()
            # nn.LogSoftmax(dim=1)
        )

    def forward(self, input):
        """Forward the discriminator."""
        out = self.layer(input)
        return out


#### Codes required by DANN ##############
class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


#### Codes required by CDAN ##############

#### Codes required by AdvSKM ##############
class Cosine_act(nn.Module):
    def __init__(self):
        super(Cosine_act, self).__init__()

    def forward(self, input):
        return torch.cos(input)


cos_act = Cosine_act()

# SASA model
class attn_network(nn.Module):
    def __init__(self, configs):
        super(attn_network, self).__init__()

        self.h_dim = configs.features_len * configs.final_out_channels
        self.self_attn_Q = nn.Sequential(nn.Linear(in_features=self.h_dim, out_features=self.h_dim),
                                         nn.ELU()
                                         )
        self.self_attn_K = nn.Sequential(nn.Linear(in_features=self.h_dim, out_features=self.h_dim),
                                         nn.LeakyReLU()
                                         )
        self.self_attn_V = nn.Sequential(nn.Linear(in_features=self.h_dim, out_features=self.h_dim),
                                         nn.LeakyReLU()
                                         )

    def forward(self, x):
        Q = self.self_attn_Q(x)
        K = self.self_attn_K(x)
        V = self.self_attn_V(x)

        return Q, K, V

    # Sparse max


class Sparsemax(nn.Module):
    """Sparsemax function."""

    def __init__(self, dim=None):
        """Initialize sparsemax activation

        Args:
            dim (int, optional): The dimension over which to apply the sparsemax function.
        """
        super(Sparsemax, self).__init__()

        self.dim = -1 if dim is None else dim

    def forward(self, input):
        """Forward function.
        Args:
            input (torch.Tensor): Input tensor. First dimension should be the batch size
        Returns:
            torch.Tensor: [batch_size x number_of_logits] Output tensor
        """
        # Sparsemax currently only handles 2-dim tensors,
        # so we reshape to a convenient shape and reshape back after sparsemax
        input = input.transpose(0, self.dim)
        original_size = input.size()
        input = input.reshape(input.size(0), -1)
        input = input.transpose(0, 1)
        dim = 1

        number_of_logits = input.size(dim)

        # Translate input by max for numerical stability
        input = input - torch.max(input, dim=dim, keepdim=True)[0].expand_as(input)

        # Sort input in descending order.
        # (NOTE: Can be replaced with linear time selection method described here:
        # http://stanford.edu/~jduchi/projects/DuchiShSiCh08.html)
        zs = torch.sort(input=input, dim=dim, descending=True)[0]
        range = torch.arange(start=1, end=number_of_logits + 1, step=1, device=input.device, dtype=input.dtype).view(1,
                                                                                                                     -1)
        range = range.expand_as(zs)

        # Determine sparsity of projection
        bound = 1 + range * zs
        cumulative_sum_zs = torch.cumsum(zs, dim)
        is_gt = torch.gt(bound, cumulative_sum_zs).type(input.type())
        k = torch.max(is_gt * range, dim, keepdim=True)[0]

        # Compute threshold function
        zs_sparse = is_gt * zs

        # Compute taus
        taus = (torch.sum(zs_sparse, dim, keepdim=True) - 1) / k
        taus = taus.expand_as(input)

        # Sparsemax
        self.output = torch.max(torch.zeros_like(input), input - taus)

        # Reshape back to original shape
        output = self.output
        output = output.transpose(0, 1)
        output = output.reshape(original_size)
        output = output.transpose(0, self.dim)

        return output

    def backward(self, grad_output):
        """Backward function."""
        dim = 1

        nonzeros = torch.ne(self.output, 0)
        sum = torch.sum(grad_output * nonzeros, dim=dim) / torch.sum(nonzeros, dim=dim)
        self.grad_input = nonzeros * (grad_output - sum.expand_as(grad_output))

        return self.grad_input


# UniOT Backbones

class classifier(nn.Module):
    def __init__(self, configs):
        super(classifier, self).__init__()
        final_out_channels = configs.final_out_channels
        if configs.isFNO:
            final_out_channels = configs.final_out_channels * 2
            final_out_channels = configs.final_out_channels + 2 * configs.fourier_modes
        if configs.isTimesNet:
            final_out_channels = configs.d_model * configs.sequence_len
        self.logits = nn.Linear(configs.features_len * final_out_channels, configs.num_classes)
        self.configs = configs

    def forward(self, x):
        predictions = self.logits(x)

        return predictions


class classifierNoBias(nn.Module):
    def __init__(self, configs):
        super(classifierNoBias, self).__init__()
        final_out_channels = configs.final_out_channels
        if configs.isFNO:
            final_out_channels = configs.final_out_channels * 2
            final_out_channels = configs.final_out_channels + 2 * configs.fourier_modes
        self.logits = nn.Linear(configs.features_len * final_out_channels, configs.num_classes, bias=False)
        self.configs = configs

    def forward(self, x):
        predictions = self.logits(x)

        return predictions


class classifier2(nn.Module):
    def __init__(self, configs):
        super(classifier2, self).__init__()
        final_out_channels = configs.final_out_channels
        if configs.isFNO:
            final_out_channels = configs.final_out_channels * 2
            final_out_channels = configs.final_out_channels + 2 * configs.fourier_modes
        if configs.isTimesNet:
            final_out_channels = nn.Linear(configs.d_model * configs.sequence_len, configs.num_classes)
        self.logits = CLS(configs)
        self.configs = configs

    def forward(self, x):
        _, predictions = self.logits(x)

        return predictions


class ProtoCLS(nn.Module):
    """
    prototype-based classifier
    L2-norm + a fc layer (without bias)
    """

    def __init__(self, int_dim, out_dim, temp=0.05):
        super(ProtoCLS, self).__init__()

        self.fc = nn.Linear(int_dim, out_dim, bias=False)
        self.tmp = temp
        self.weight_norm()

    def forward(self, x):
        x = F.normalize(x)
        x = self.fc(x) / self.tmp
        return x

    def weight_norm(self):
        w = self.fc.weight.data
        norm = w.norm(p=2, dim=1, keepdim=True)
        self.fc.weight.data = w.div(norm.expand_as(w))


'''class CLS(nn.Module):
    """
    a classifier made up of projection head and prototype-based classifier
    """

    def __init__(self, configs, temp=0.05):
        super(CLS, self).__init__()
        self.projection_head = nn.Sequential(
            nn.Linear(configs.features_len * configs.final_out_channels, configs.features_len * configs.final_out_channels//2),
            nn.ReLU(inplace=True),
            nn.Linear(configs.features_len * configs.final_out_channels//2, configs.final_out_channels))
        self.ProtoCLS = ProtoCLS(configs.final_out_channels, configs.num_classes, temp)

    def forward(self, x):
        before_lincls_feat = self.projection_head(x)
        after_lincls = self.ProtoCLS(before_lincls_feat)
        return before_lincls_feat, after_lincls'''


class CLS(nn.Module):
    """
    a classifier made up of projection head and prototype-based classifier
    """

    def __init__(self, configs, temp=0.05):
        super(CLS, self).__init__()
        final_out_channels = configs.final_out_channels
        if configs.isFNO:
            final_out_channels = configs.final_out_channels * 2
            final_out_channels = configs.final_out_channels + 2 * configs.fourier_modes
            # print(final_out_channels)
        self.projection_head = nn.Sequential(
            nn.Linear(configs.features_len * final_out_channels, configs.features_len * final_out_channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(configs.features_len * final_out_channels // 2, final_out_channels))
        self.ProtoCLS = ProtoCLS(final_out_channels, configs.num_classes, temp)

    def forward(self, x):
        before_lincls_feat = self.projection_head(x)
        after_lincls = self.ProtoCLS(before_lincls_feat)
        return before_lincls_feat, after_lincls


class MemoryQueue(nn.Module):
    def __init__(self, feat_dim, batchsize, n_batch, T=0.05):
        super(MemoryQueue, self).__init__()
        self.feat_dim = feat_dim
        self.batchsize = batchsize
        self.T = T

        # init memory queue
        self.queue_size = self.batchsize * n_batch
        self.register_buffer('mem_feat', torch.zeros(self.queue_size, feat_dim))
        self.register_buffer('mem_id', torch.zeros((self.queue_size), dtype=int))
        self.mem_feat = self.mem_feat.cuda()
        self.mem_id = self.mem_id.cuda()

        # write pointer
        self.next_write = 0

    def forward(self, x):
        """
        obtain similarity between x and the features stored in memory queue指针 英语
        """
        out = torch.mm(x, self.mem_feat.t()) / self.T
        return out

    def get_nearest_neighbor(self, anchors, id_anchors=None):
        """
        get anchors' nearest neighbor in memory queue
        """
        # compute similarity first
        feat_mat = self.forward(anchors)

        # assign the similarity between features of the same sample with -1/T
        if id_anchors is not None:
            A = id_anchors.reshape(-1, 1).repeat(1, self.mem_id.size(0))
            B = self.mem_id.reshape(1, -1).repeat(id_anchors.size(0), 1)
            mask = torch.eq(A, B)
            id_mask = torch.nonzero(mask)
            temp = id_mask[:, 1]
            feat_mat[:, temp] = -1 / self.T

        # obtain neighbor's similarity value and corresponding feature
        values, indices = torch.max(feat_mat, 1)
        nearest_feat = torch.zeros((anchors.size(0), self.feat_dim)).cuda()
        for i in range(anchors.size(0)):
            nearest_feat[i] = self.mem_feat[indices[i], :]
        return values, nearest_feat

    def update_queue(self, features, ids):
        """
        update memory queue
        """
        w_ids = torch.arange(self.next_write, self.next_write + self.batchsize).cuda()
        self.mem_feat.index_copy_(0, w_ids, features.data)
        self.mem_id.index_copy_(0, w_ids, ids.data)
        self.mem_feat = F.normalize(self.mem_feat)

        # update write pointer
        self.next_write += self.batchsize
        if self.next_write == self.queue_size:
            self.next_write = 0

    def random_sample(self, size):
        """
        sample some features from memory queue randomly
        """
        id_t = torch.floor(torch.rand(size) * self.mem_feat.size(0)).long().cuda()
        sample_feat = self.mem_feat[id_t]
        return sample_feat


class ClassMemoryQueue(nn.Module):
    def __init__(self, feat_dim, num_classes, N, T=0.05):
        super(ClassMemoryQueue, self).__init__()
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.N = N
        self.T = T

        # Memory for each class
        self.register_buffer('mem_feat', torch.zeros(num_classes, N, feat_dim))
        self.register_buffer('mem_count',
                             torch.zeros(num_classes, dtype=torch.int64))  # Keeps track of the count per class

    def forward(self, x):
        """
        Obtain similarity between x and the features stored in memory queue (all classes).
        """
        mem_feat_flat = self.mem_feat.view(-1, self.feat_dim)  # Flatten the memory queue
        out = torch.mm(x, mem_feat_flat.t()) / self.T
        return out

    def update_queue(self, features, labels):
        """
        Updates the memory queue for each class using masks to avoid direct iteration over labels.
        Args:
            features (torch.Tensor): Input features of shape (batch_size, feat_dim).
            labels (torch.Tensor): Class labels of shape (batch_size).
        """
        with torch.no_grad():
            for i in range(self.num_classes):
                # Create mask for current class
                mask = labels == i
                if mask.any():
                    # Extract features and count for the current class
                    class_features = features[mask]  # Features for the current class
                    class_count = class_features.size(0)  # Number of features for this class
                    current_count = self.mem_count[i].item()  # Current memory count for this class

                    if current_count + class_count <= self.N:
                        print("in")
                        # If enough space, add all features
                        self.mem_feat[i, current_count:current_count + class_count] = class_features
                        self.mem_count[i] += class_count
                    else:
                        # If not enough space, add as many as possible, replace oldest features
                        space_needed = self.N - min(self.N, class_count)
                        self.mem_feat[i, space_needed:] = class_features[:min(self.N, class_count)]
                        self.mem_count[i] = min(self.N, current_count + class_count)

    def get_class_features(self, class_idx):
        """
        Get features stored for a specific class.
        """
        count = self.mem_count[class_idx].item()
        return self.mem_feat[class_idx, :count]

    def compute_distances(self, target_features):
        """
        Compute distances between each target point and the closest memory point for each class.

        Args:
            target_features (torch.Tensor): Tensor of size (Nt, feat_dim), where Nt is the number of target points.

        Returns:
            torch.Tensor: A tensor of size (Nt, num_classes), where each element [i, k] is the distance
                          between target point i and the closest memory point in class k.
        """
        Nt = target_features.size(0)
        distances = torch.zeros(Nt, self.num_classes).to(target_features.device)

        for k in range(self.num_classes):
            class_features = self.get_class_features(k)  # Features for class k (size: [count_k, feat_dim])
            if class_features.size(0) == 0:
                # If there are no features for class k, set distance to a large value
                distances[:, k] = float('inf')
                continue

            # Compute distances between each target feature and all class features
            dists = torch.cdist(target_features.cpu(), class_features.cpu())  # Shape: (Nt, count_k)

            # Take the minimum distance for each target point
            distances[:, k] = dists.min(dim=1).values  # Shape: (Nt,)

        return distances

    def is_memory_full(self):
        """
        Check if the memory is full for each class.

        Returns:
            torch.Tensor: A tensor of boolean values indicating whether each class's memory is full.
        """
        return self.mem_count >= self.N