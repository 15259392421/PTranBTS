from copy import deepcopy
import torch.nn.functional as F
import torch.nn.functional
from torch import nn
import torch
import math
softmax_helper = lambda x: F.softmax(x, 1)

class InitWeights_He(object):
    def __init__(self, neg_slope=1e-2):
        self.neg_slope = neg_slope

    def __call__(self, module):
        if isinstance(module, nn.Conv3d) or isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d) or isinstance(module, nn.ConvTranspose3d):
            module.weight = nn.init.kaiming_normal_(module.weight, a=self.neg_slope)
            if module.bias is not None:
                module.bias = nn.init.constant_(module.bias, 0)

class ConvDropoutNormNonlin(nn.Module):
    def __init__(self, input_channels, output_channels, conv_kwargs=None):
        super(ConvDropoutNormNonlin, self).__init__()
        if conv_kwargs is None:
            conv_kwargs = {'stride': 1, 'dilation': 1, 'bias': True, 'kernel_size': [3, 3, 3], 'padding': [1, 1, 1]}

        self.nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        self.nonlin = nn.LeakyReLU
        self.dropout_op = nn.Dropout3d
        self.dropout_op_kwargs = {'p': 0, 'inplace': True}
        self.norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        self.conv_kwargs = conv_kwargs
        self.conv_op = nn.Conv3d
        self.norm_op = nn.InstanceNorm3d

        self.conv = self.conv_op(input_channels, output_channels, **self.conv_kwargs)
        if self.dropout_op is not None and self.dropout_op_kwargs['p'] is not None and self.dropout_op_kwargs[
            'p'] > 0:
            self.dropout = self.dropout_op(**self.dropout_op_kwargs)
        else:
            self.dropout = None
        self.instnorm = self.norm_op(output_channels, **self.norm_op_kwargs)
        self.lrelu = self.nonlin(**self.nonlin_kwargs)

    def forward(self, x):
        x = self.conv(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return self.lrelu(self.instnorm(x))


class BasicConv(nn.Module):
    def __init__(self, in_channels, conv_in_channels, conv_out_channels):
        super(BasicConv, self).__init__()
        self.MLP_conv_kwargs = {'stride': 1, 'dilation': 1, 'bias': True, 'kernel_size': [1, 1, 1]}
        self.MLP = ConvDropoutNormNonlin(in_channels, conv_in_channels, self.MLP_conv_kwargs)
        self.Conv = ConvDropoutNormNonlin(conv_in_channels, conv_out_channels)

    def forward(self, x):
        x = self.MLP(x)
        x = self.Conv(x)
        return x


class BasicTrans(nn.Module):
    def __init__(self, in_channels, Trans_channels, r, heads):
        super(BasicTrans, self).__init__()
        self.depthwise_conv = nn.Conv3d(
            in_channels, in_channels, stride=r, kernel_size=r, groups=in_channels, bias=False
        )
        self.bn = nn.BatchNorm3d(in_channels)

        self.head_dim = Trans_channels // heads
        self.scale = self.head_dim**-0.5
        self.num_heads = heads
        # qkv
        self.qkv = nn.Conv3d(in_channels, Trans_channels * 3, kernel_size=1, bias=False)

        self.norm = nn.GroupNorm(num_groups=1, num_channels=Trans_channels)
        self.conv_trans = nn.ConvTranspose3d(
            Trans_channels, Trans_channels, kernel_size=r, stride=r, groups=Trans_channels
        )

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.bn(x)

        B, C, H, W, Z = x.shape
        q, k, v = (
            self.qkv(x)
            .view(B, self.num_heads, -1, H * W * Z)
            .split([self.head_dim, self.head_dim, self.head_dim], dim=2)
        )
        attn = (self.scale * q.transpose(-2, -1) @ k).softmax(-1)
        x = (v @ attn.transpose(-2, -1)).view(B, -1, H, W, Z)

        x = self.conv_trans(x)
        x = self.norm(x)
        return x




class StackedConvLayers(nn.Module):
    def __init__(self, input_feature_channels, output_feature_channels, num_convs,
                conv_kwargs=None, first_stride=None, basic_block=ConvDropoutNormNonlin):

        self.input_channels = input_feature_channels
        self.output_channels = output_feature_channels

        if conv_kwargs is None:
            conv_kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1, 'dilation': 1, 'bias': True}

        self.conv_kwargs = conv_kwargs


        if first_stride is not None:
            self.conv_kwargs_first_conv = deepcopy(conv_kwargs)
            self.conv_kwargs_first_conv['stride'] = first_stride
        else:
            self.conv_kwargs_first_conv = conv_kwargs

        super(StackedConvLayers, self).__init__()
        self.blocks = nn.Sequential(
            *([basic_block(input_feature_channels, output_feature_channels,
                           self.conv_kwargs_first_conv)] +
              [basic_block(output_feature_channels, output_feature_channels,
                           self.conv_kwargs) for _ in range(num_convs - 1)]))

    def forward(self, x):
        return self.blocks(x)



class Cross_Modality_Conv(nn.Module):
    def __init__(self, in_channels, branchs_in=4):
        super(Cross_Modality_Conv, self).__init__()
        MLP_conv_kwargs = {'stride': 1, 'dilation': 1, 'bias': True, 'kernel_size': [1, 1, 1]}
        self.MLP = ConvDropoutNormNonlin(in_channels*branchs_in, in_channels, MLP_conv_kwargs)
        # DW_conv_kwargs = {'stride': 1, 'dilation': 1, 'bias': False, 'kernel_size': [3, 3, 3], 'padding': [1, 1, 1],'groups':in_channels * branchs_in}
        # self.DW_Conv = ConvDropoutNormNonlin(in_channels*branchs_in, in_channels*branchs_in, DW_conv_kwargs)


    def forward(self, x):
        # x = self.DW_Conv(x)
        x = self.MLP(x)
        return x

class BTSS_in(nn.Module):
    def __init__(self, input_features, output_features, branchs=4, conv_kwargs=None, num_conv_per_stage=2):
        super(BTSS_in, self).__init__()
        self.output_features = output_features
        self.branchs = branchs
        if self.branchs == 4:
            self.DWconv_t1 = StackedConvLayers(input_features, output_features, num_conv_per_stage,
                              conv_kwargs)
            self.DWconv_t1ce = StackedConvLayers(input_features, output_features, num_conv_per_stage,
                              conv_kwargs)
            self.DWconv_t2 = StackedConvLayers(input_features, output_features, num_conv_per_stage,
                              conv_kwargs)
            self.DWconv_flair = StackedConvLayers(input_features, output_features, num_conv_per_stage,
                              conv_kwargs)
        elif self.branchs == 1:
            self.DWconv = StackedConvLayers(4, output_features, num_conv_per_stage,
                                               conv_kwargs)


    def forward(self, x):
        if self.branchs == 4:
            t1, t1ce, t2, flair = x[:,0,:,:,:].unsqueeze(1), x[:,1,:,:,:].unsqueeze(1), x[:,2,:,:,:].unsqueeze(1), x[:,3,:,:,:].unsqueeze(1)
            t1 = self.DWconv_t1(t1)
            t1ce = self.DWconv_t1ce(t1ce)
            t2 = self.DWconv_t2(t2)
            flair = self.DWconv_flair(flair)
            x = torch.cat([t1, t1ce, t2, flair], dim=1)
        elif self.branchs == 1:
            x = self.DWconv(x)
        return x

class SegHead(nn.Module):
    def __init__(self, in_channel=32,branchs_out=3, num_classes=1, conv_kwargs = None):
        super(SegHead, self).__init__()
        if branchs_out == 3:
            self.CMC = Cross_Modality_Conv(in_channels=in_channel, branchs_in=4)
        elif branchs_out == 1:
            self.CMC = Cross_Modality_Conv(in_channels=in_channel, branchs_in=1)
        self.dec_0 = nn.Sequential(StackedConvLayers(in_channel * 2, in_channel, 1, conv_kwargs),
                              StackedConvLayers(in_channel, in_channel, 1, conv_kwargs))
        self.Seg =nn.Conv3d(in_channel, num_classes,1, 1, 0, 1, 1, False)

    def forward(self, x, skip_feature):
        skip_feature = self.CMC(skip_feature)
        x = torch.cat((x, skip_feature), dim=1)
        x = self.dec_0(x)
        x = self.Seg(x)
        return x

class BTSS_out(nn.Module):
    def __init__(self, in_channel=32, branchs=3, num_classes = 3, conv_kwargs = None):
        super(BTSS_out, self).__init__()
        self.branchs = branchs
        if self.branchs == 3:
            self.num_classes = num_classes
            self.Seglist = []
            for i in range(num_classes):
                self.Seglist.append(SegHead(in_channel,conv_kwargs=conv_kwargs))
            self.Seglist = nn.ModuleList(self.Seglist)
        elif self.branchs == 1:
            self.Seghead = SegHead(in_channel, num_classes=3,branchs_out=self.branchs, conv_kwargs=conv_kwargs)



    def forward(self, x ,skip_feature):
        if self.branchs == 3:
            class_out = []
            for i in range(self.num_classes):
                Segout = self.Seglist[i](x, skip_feature)
                class_out.append(Segout)
            out = torch.cat(class_out,dim=1)
        elif self.branchs == 1:
            out = self.Seghead(x, skip_feature)
        return out


class BasicBlock(nn.Module):
    def __init__(self, in_channels, conv_in_channels, conv_out_channels, Trans_channels, r, heads):
        super(BasicBlock, self).__init__()
        self.Trans_channels = Trans_channels
        self.conv_head = BasicConv(in_channels=in_channels, conv_in_channels=conv_in_channels, conv_out_channels=conv_out_channels)
        if self.Trans_channels != 0:
            self.trans_head = BasicTrans(in_channels=in_channels, Trans_channels=Trans_channels, r=r, heads=heads)
        self.MLP_conv_kwargs = {'stride': 1, 'dilation': 1, 'bias': False, 'kernel_size': [1, 1, 1],
                                'padding': 0}
        self.MLP = ConvDropoutNormNonlin(conv_out_channels + Trans_channels, conv_out_channels + Trans_channels, self.MLP_conv_kwargs)

    def forward(self, x):
        conv_f = self.conv_head(x)
        if self.Trans_channels != 0:
            trans_f = self.trans_head(x)
            out = self.MLP(torch.cat([conv_f, trans_f], dim=1)) + x
        else:
            out = conv_f + x
        return out


class DownConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownConvLayer, self).__init__()
        self.Dconv_conv_kwargs = {'stride': 2, 'dilation': 1, 'bias': True, 'kernel_size': [3, 3, 3],
                                'padding': [1, 1, 1]}
        self.Dconv = ConvDropoutNormNonlin(in_channels, out_channels, self.Dconv_conv_kwargs)

    def forward(self, x):
        x = self.Dconv(x)
        return x

class EncoderBasicBlock(nn.Module):
    def __init__(self, in_channels, r, heads, conv_proportion=0.8):
        super(EncoderBasicBlock, self).__init__()
        self.conv_proportion = conv_proportion
        self.trans_proportion = 1 - conv_proportion + 0.00001
        self.block = BasicBlock(in_channels=in_channels, conv_in_channels=int(in_channels*self.conv_proportion),
                                    conv_out_channels=int(in_channels*self.conv_proportion),
                                Trans_channels=int(in_channels*self.trans_proportion), r=r, heads=heads)

    def forward(self, x):
        x = self.block(x)
        return x

class DecoderBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, r, heads, conv_proportion=0.8):
        super(DecoderBasicBlock, self).__init__()
        self.Conv = ConvDropoutNormNonlin(in_channels, out_channels)
        self.conv_proportion = conv_proportion
        self.trans_proportion = 1 - conv_proportion + 0.00001
        self.block = BasicBlock(in_channels=out_channels, conv_in_channels=int(out_channels*self.conv_proportion),
                                    conv_out_channels=int(out_channels*self.conv_proportion),
                                Trans_channels=int(out_channels*self.trans_proportion), r=r, heads=heads)

    def forward(self, x, skip_x):
        x = torch.cat((x, skip_x), dim=1)
        x = self.Conv(x)
        x = self.block(x)
        return x



class ShapePriorsExtract(nn.Module):
    def __init__(self, in_channel, r, N = 8):
        super(ShapePriorsExtract, self).__init__()
        self.depthwise_conv = nn.Conv3d(
            in_channel, in_channel, stride=r, kernel_size=r, groups=in_channel, bias=False
        )
        self.bn = nn.BatchNorm3d(in_channel)

        self.Conv_Fd = ConvDropoutNormNonlin(in_channel, N)
        self.q = nn.Conv3d(N, N, kernel_size=1, bias=False)
        self.kv = nn.Conv3d(2*N, 2*N, kernel_size=1, bias=False)
        self.num_heads = 1
        self.head_dim = N // self.num_heads
        self.scale = self.head_dim**-0.5
        self.norm1 = nn.GroupNorm(num_groups=1, num_channels=N)

        self.norm2 = nn.GroupNorm(num_groups=1, num_channels=N)

        self.conv_trans = nn.ConvTranspose3d(
            N, N, kernel_size=r, stride=r
        )

    def forward(self, original_shape_prior, decoder_feature):
        Fd = self.depthwise_conv(decoder_feature)
        Fd = self.bn(Fd)
        Fd = self.Conv_Fd(Fd)

        B, C, H, W, Z = original_shape_prior.shape

        k, v = (
            self.kv(original_shape_prior)
            .view(B, self.num_heads, -1, H * W * Z)
            .split([self.head_dim, self.head_dim], dim=2)
        )
        B, C, H, W, Z = Fd.shape

        q = (
            self.q(Fd)
            .view(B, self.num_heads, -1, H * W * Z)
        )

        attn = (self.scale * q.transpose(-2, -1) @ k).softmax(-1)
        x = (v @ attn.transpose(-2, -1)).view(B, -1, H, W, Z)
        x = self.norm1(x)

        x = self.conv_trans(x)
        x = self.norm2(x)
        return x

class GlobalPriorsEmbedding(nn.Module):
    def __init__(self, in_channel, N = 8):
        super(GlobalPriorsEmbedding, self).__init__()
        self.Conv_Fs1 = ConvDropoutNormNonlin(in_channel, in_channel)
        # self.Conv_Fs2 = ConvDropoutNormNonlin(in_channel, in_channel)
        self.N = N
        self.TSconv = nn.ConvTranspose3d(3*N, N, [2, 2, 2], [2, 2, 2], bias=False)
        self.MLP = nn.Conv3d(in_channels=3*N, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)
        self.num_heads = 1



    def forward(self, skip_feature, refined_shape_prior):

        class_feature = torch.matmul(skip_feature.flatten(2), refined_shape_prior.flatten(2).transpose(-1, -2))
        # scale
        class_feature = class_feature / math.sqrt(self.N)
        class_feature = class_feature.softmax(-1)

        class_feature = torch.einsum("ijk, iklhw->ijlhw", class_feature, refined_shape_prior)
        class_feature = skip_feature + class_feature

        return class_feature



class LocalPriorsEmbedding(nn.Module):
    def __init__(self, in_channel, N = 24):
        super(LocalPriorsEmbedding, self).__init__()
        self.Conv_Fs = ConvDropoutNormNonlin(in_channel, N)
        # self.conv_d = nn.Conv3d(2*N, in_channel, kernel_size=1, stride=1, padding=0)
        # self.conv_h = nn.Conv3d(2*N, in_channel, kernel_size=1, stride=1, padding=0)
        # self.conv_w = nn.Conv3d(2*N, in_channel, kernel_size=1, stride=1, padding=0)


    def forward(self, skip_feature, original_shape_prior):
        Fs = self.Conv_Fs(skip_feature)
        refined_shape_prior = torch.cat((Fs, original_shape_prior), dim=1)
        Fs_e = skip_feature
        return Fs_e, refined_shape_prior

class PyramidPriorsModule(nn.Module):
    def __init__(self, in_channel, r, N = 8):
        super(PyramidPriorsModule, self).__init__()
        self.SPE = ShapePriorsExtract(in_channel=in_channel, r=r, N=N)
        self.GPE = GlobalPriorsEmbedding(in_channel=in_channel, N=N)
        self.LPE = LocalPriorsEmbedding(in_channel=in_channel, N=N)



    def forward(self, skip_feature, original_shape_prior, decoder_feature):
        refined_shape_prior = self.SPE(original_shape_prior, decoder_feature)
        Fs_e = self.GPE(skip_feature, refined_shape_prior)
        Fs_e, refined_shape_prior = self.LPE(Fs_e, refined_shape_prior)
        return Fs_e, refined_shape_prior

class Encoder(nn.Module):
    def __init__(
        self,
        channels=(24, 64, 128, 256, 320, 320),
        blocks=(1, 1, 1, 1, 1),
        heads=(1, 2, 2, 4, 8),
        r=(4, 2, 2, 1, 1),
        branchs_in=4,
        conv_proportion=0.8
    ):
        super(Encoder, self).__init__()

        self.B_in = BTSS_in(input_features=1, output_features=channels[0],branchs=branchs_in)
        # self.CMC_0 = Cross_Modality_Conv(in_channels=channels[0],branchs_in=branchs_in)
        #
        # self.DWconv1 = DownConvLayer(in_channels=channels[0], out_channels=channels[1])
        MLP_conv_kwargs = {'stride': 1, 'dilation': 1, 'bias': True, 'kernel_size': [1, 1, 1]}
        self.MLP = ConvDropoutNormNonlin(channels[0]*4, channels[0]*2, MLP_conv_kwargs)
        self.DWconv1 = DownConvLayer(in_channels=channels[0]*2, out_channels=channels[1])
        self.DWconv2 = DownConvLayer(in_channels=channels[1], out_channels=channels[2])
        self.DWconv3 = DownConvLayer(in_channels=channels[2], out_channels=channels[3])
        self.DWconv4 = DownConvLayer(in_channels=channels[3], out_channels=channels[4])
        self.DWconv5 = DownConvLayer(in_channels=channels[4], out_channels=channels[5])

        block = []
        for _ in range(blocks[0]):
            block.append(EncoderBasicBlock(conv_proportion=conv_proportion, in_channels=channels[1], r=r[0], heads=heads[0]))
        self.block1 = nn.Sequential(*block)

        block = []
        for _ in range(blocks[1]):
            block.append(EncoderBasicBlock(conv_proportion=conv_proportion, in_channels=channels[2], r=r[1], heads=heads[1]))
        self.block2 = nn.Sequential(*block)

        block = []
        for _ in range(blocks[2]):
            block.append(EncoderBasicBlock(conv_proportion=conv_proportion, in_channels=channels[3], r=r[2], heads=heads[2]))
        self.block3 = nn.Sequential(*block)

        block = []
        for _ in range(blocks[3]):
            block.append(EncoderBasicBlock(conv_proportion=conv_proportion, in_channels=channels[4], r=r[3], heads=heads[3]))
        self.block4 = nn.Sequential(*block)

        block = []
        for _ in range(blocks[4]):
            block.append(EncoderBasicBlock(conv_proportion=conv_proportion, in_channels=channels[5], r=r[4], heads=heads[4]))
        self.block5 = nn.Sequential(*block)

    def forward(self, x):
        hidden_states_out = []
        x = self.B_in(x)
        hidden_states_out.append(x)
        x = self.MLP(x)

        x = self.DWconv1(x)
        x = self.block1(x)
        hidden_states_out.append(x)
        x = self.DWconv2(x)
        x = self.block2(x)
        hidden_states_out.append(x)
        x = self.DWconv3(x)
        x = self.block3(x)
        hidden_states_out.append(x)
        x = self.DWconv4(x)
        x = self.block4(x)
        hidden_states_out.append(x)
        x = self.DWconv5(x)
        x = self.block5(x)
        return x, hidden_states_out


class Decoder(nn.Module):
    def __init__(
        self,
        num_classes=3,
        channels=(24, 64, 128, 256, 320, 320),
        blocks=(1, 1, 1, 1, 1),
        heads=(1, 2, 2, 4, 8),
        r=(4, 2, 2, 1, 1),
        branchs_out=3,
        AgN=8,
        conv_proportion=0.8
    ):
        super(Decoder, self).__init__()
        # self.do_ds = deep_supervision

        self.TSconv4 = nn.ConvTranspose3d(channels[5], channels[4], [2, 2, 2], [2, 2, 2], bias=False)
        self.TSconv3 = nn.ConvTranspose3d(channels[4], channels[3], [2, 2, 2], [2, 2, 2], bias=False)
        self.TSconv2 = nn.ConvTranspose3d(channels[3], channels[2], [2, 2, 2], [2, 2, 2], bias=False)
        self.TSconv1 = nn.ConvTranspose3d(channels[2], channels[1], [2, 2, 2], [2, 2, 2], bias=False)
        self.TSconv0 = nn.ConvTranspose3d(channels[1], channels[0], [2, 2, 2], [2, 2, 2], bias=False)

        self.block1 = DecoderBasicBlock(conv_proportion=conv_proportion, in_channels=channels[1] * 2, out_channels=channels[1], r=r[0], heads=heads[0])
        self.block2 = DecoderBasicBlock(conv_proportion=conv_proportion, in_channels=channels[2] * 2, out_channels=channels[2], r=r[1], heads=heads[1])
        self.block3 = DecoderBasicBlock(conv_proportion=conv_proportion, in_channels=channels[3] * 2, out_channels=channels[3], r=r[2], heads=heads[2])
        self.block4 = DecoderBasicBlock(conv_proportion=conv_proportion, in_channels=channels[4] * 2, out_channels=channels[4], r=r[3], heads=heads[3])

        self.B_out = BTSS_out(in_channel=channels[0], num_classes=3,branchs=branchs_out)


        self.seg_output320 = nn.Conv3d(channels[-2], num_classes, 1, 1, 0, 1, 1, False)
        self.seg_output256 = nn.Conv3d(channels[-3], num_classes, 1, 1, 0, 1, 1, False)
        self.seg_output128 = nn.Conv3d(channels[-4], num_classes, 1, 1, 0, 1, 1, False)
        self.seg_output64 = nn.Conv3d(channels[-5], num_classes, 1, 1, 0, 1, 1, False)

        self.learnable_shape_prior = nn.Parameter(torch.randn(1, 2*AgN, 4, 4, 4))
        self.PPM4 = PyramidPriorsModule(channels[4], r=r[3], N=AgN)
        self.PPM3 = PyramidPriorsModule(channels[3], r=r[2], N=AgN)
        self.PPM2 = PyramidPriorsModule(channels[2], r=r[1], N=AgN)
        self.PPM1 = PyramidPriorsModule(channels[1], r=r[0], N=AgN)

    def forward(self, x, hidden_states_out):
        seg_outputs = []
        B = x.size()[0]
        learnable_shape_prior = self.learnable_shape_prior.repeat(B, 1, 1, 1, 1)
        x = self.TSconv4(x)
        Fs_e4, refined_shape_prior = self.PPM4(skip_feature=hidden_states_out[-1],
                                              original_shape_prior=learnable_shape_prior, decoder_feature=x)
        x = self.block4(x, Fs_e4)
        seg_outputs.append(self.seg_output320(x))
        x = self.TSconv3(x)
        Fs_e3, refined_shape_prior = self.PPM3(skip_feature=hidden_states_out[-2],
                                              original_shape_prior=refined_shape_prior, decoder_feature=x)
        x = self.block3(x, Fs_e3)
        seg_outputs.append(self.seg_output256(x))
        x = self.TSconv2(x)
        Fs_e2, refined_shape_prior = self.PPM2(skip_feature=hidden_states_out[-3],
                                              original_shape_prior=refined_shape_prior, decoder_feature=x)
        x = self.block2(x, Fs_e2)
        seg_outputs.append(self.seg_output128(x))
        x = self.TSconv1(x)
        Fs_e1, refined_shape_prior = self.PPM1(skip_feature=hidden_states_out[-4],
                                              original_shape_prior=refined_shape_prior, decoder_feature=x)
        x = self.block1(x, Fs_e1)
        seg_outputs.append(self.seg_output64(x))
        x = self.TSconv0(x)
        x = self.B_out(x, hidden_states_out[-5])
        seg_outputs.append(x)
        # seg_outputs = []
        # x = self.TSconv4(x)
        # x = self.block4(x, hidden_states_out[-1])
        # seg_outputs.append(self.seg_output320(x))
        # x = self.TSconv3(x)
        # x = self.block3(x, hidden_states_out[-2])
        # seg_outputs.append(self.seg_output256(x))
        # x = self.TSconv2(x)
        # x = self.block2(x, hidden_states_out[-3])
        # seg_outputs.append(self.seg_output128(x))
        # x = self.TSconv1(x)
        # x = self.block1(x, hidden_states_out[-4])
        # seg_outputs.append(self.seg_output64(x))
        # x = self.TSconv0(x)
        # x = self.B_out(x, hidden_states_out[-5])
        # seg_outputs.append(x)
        return seg_outputs

class PHtransBTS(nn.Module):
    def __init__(
        self,
        channels=(24, 80, 160, 320, 400, 400),
        blocks=(1, 1, 1, 1, 1),
        heads=(1, 2, 2, 4, 8),
        r=(4, 2, 2, 1, 1),
        deep_supervision=True,
        branch_in=4,
        branch_out=3,
        AgN=24,
        conv_proportion = 0.8
    ):
        super(PHtransBTS, self).__init__()
        self.do_ds = deep_supervision
        self.Encoder = Encoder(
            channels=channels,
            blocks=blocks,
            heads=heads,
            r=r,
            branchs_in=branch_in,
            conv_proportion=conv_proportion
        )
        self.Decoder = Decoder(
            channels=channels,
            blocks=blocks,
            heads=heads,
            r=r,
            branchs_out=branch_out,
            AgN=AgN,
            conv_proportion=conv_proportion
            # deep_supervision=deep_supervision
        )
        self.apply(InitWeights_He(1e-2))



    def forward(self, x):
        embeding, hidden_states_out = self.Encoder(x)
        seg_outputs = self.Decoder(embeding, hidden_states_out)

        upscale_logits_ops = []
        for usl in range(4):
            upscale_logits_ops.append(lambda x: x)
        if self.do_ds:
            return tuple([seg_outputs[-1]] + [i(j) for i, j in
                     zip(list(upscale_logits_ops)[::-1], seg_outputs[:-1][::-1])])
        else:
            return seg_outputs[-1]
        # return x



if __name__ == "__main__":
    x = torch.randn(size=(1, 4, 128, 128, 128)).cuda()
    # with torch.no_grad():
    model = PHtransBTS(channels=(24, 80, 160, 320, 400, 400),
                   blocks=(1, 1, 1, 2, 3),
                   heads=(1, 2, 2, 4, 8),
                   r=(4, 2, 1, 1, 1),
                   deep_supervision=True,
                   branch_in=1, branch_out=1
                   , AgN=8,conv_proportion=0.8).cuda()
    model.eval()
    y = model(x)
    print(y)