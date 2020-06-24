import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchvision import models
import numpy as np
import random

###########################################
# VGG Encoder 
###########################################
class VGG19_BN(nn.Module):
    """VGG19_BN features without the last (5th) pooling layer."""
    cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512]
    
    def __init__(self, in_channels=15):
        super(VGG19_BN, self).__init__()
        self.features = self.make_layers(self.cfg, in_channels, batch_norm=True)
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    @staticmethod
    def make_layers(cfg, in_channels, batch_norm=False):
        layers = []
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)
    
###########################################
# Content Encoder
###########################################
class ContentEncoder(nn.Module):
    def __init__(self, num_adain_params, g1_out_size=512, string_len=17, num_chars=26, emb_size=5):
        super(ContentEncoder, self).__init__()
        self.embedding = nn.Embedding(num_chars+1, emb_size, padding_idx=0)
        self.g1 = MultiLayerPerceptron(emb_size, g1_out_size)
        self.g2 = MultiLayerPerceptron(emb_size * string_len, num_adain_params)
    
    def forward(self, t):
        e = self.embedding(t)
        # g1 takes individual character
        Fc = self.g1(e.view((-1, e.shape[-1]))).view(*e.shape[:2], -1)
        # g2 takes whole words
        fc = self.g2(e)
        return Fc, fc


class MultiLayerPerceptron(nn.Module):
    def __init__(self, in_size, out_size, h_size=256):
        super(MultiLayerPerceptron, self).__init__()
        layers = []
        sizes = [in_size, h_size, h_size, out_size]
        n = len(sizes) - 1
        for k in range(n):
            layers += [nn.Linear(sizes[k], sizes[k+1])]
            if k != n - 1:
                layers += [nn.BatchNorm1d(sizes[k+1]), nn.ReLU()]
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.model(x)
        return x

###########################################
# Decoder (generator)
###########################################
# Reference: https://github.com/eriklindernoren/PyTorch-GAN/blob/a163b82beff3d01688d8315a3fd39080400e7c01/implementations/munit/models.py

class Decoder(nn.Module):
    def __init__(self, out_channels=1, dim=64, n_residual=2, n_upsample=4):
        super(Decoder, self).__init__()
        
        layers = []
        dim = dim * 2 ** n_upsample
        # Residual blocks
        for _ in range(n_residual):
            layers += [ResidualBlock(dim, norm="adain")]
            
        # Upsampling
        for _ in range(n_upsample):
            layers += [
                nn.Upsample(scale_factor=2),
                nn.Conv2d(dim, dim // 2, 5, stride=1, padding=2),
                LayerNorm(dim // 2),
                nn.ReLU(inplace=True),
            ]
            dim = dim // 2

        # Output layer
        layers += [nn.ReflectionPad2d(3), nn.Conv2d(dim, out_channels, 7), nn.Tanh()]

        self.model = nn.Sequential(*layers)
    
    def get_num_adain_params(self):
        """Return the number of AdaIN parameters needed by the model"""
        num_adain_params = 0
        for m in self.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                num_adain_params += 2 * m.num_features
        return num_adain_params
    
    def assign_adain_params(self, adain_params):
        """Assign the adain_params to the AdaIN layers in model"""
        for m in self.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                # Extract mean and std predictions
                mean = adain_params[:, : m.num_features]
                std = adain_params[:, m.num_features : 2 * m.num_features]
                # Update bias and weight
                m.bias = mean.contiguous().view(-1)
                m.weight = std.contiguous().view(-1)
                # Move pointer
                if adain_params.size(1) > 2 * m.num_features:
                    adain_params = adain_params[:, 2 * m.num_features :]
    
    def forward(self, F, adain_params):
        # Update AdaIN parameters by MLP prediction based off style code
        self.assign_adain_params(adain_params)
        img = self.model(F)
        return img


class ResidualBlock(nn.Module):
    def __init__(self, features, norm="in"):
        super(ResidualBlock, self).__init__()

        norm_layer = AdaptiveInstanceNorm2d if norm == "adain" else nn.InstanceNorm2d

        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(features, features, 3),
            norm_layer(features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(features, features, 3),
            norm_layer(features),
        )

    def forward(self, x):
        return x + self.block(x)

    
class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # weight and bias are dynamically assigned
        self.weight = None
        self.bias = None
        # just dummy buffers, not used
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def forward(self, x):
        assert (
            self.weight is not None and self.bias is not None
        ), "Please assign weight and bias before calling AdaIN!"
        b, c, h, w = x.size()
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)

        # Apply instance norm
        x_reshaped = x.contiguous().view(1, b * c, h, w)

        out = F.batch_norm(x_reshaped, running_mean, running_var, self.weight, self.bias, True, self.momentum, self.eps)
        return out.view(b, c, h, w)

    def __repr__(self):
        return self.__class__.__name__ + "(" + str(self.num_features) + ")"


class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        mean = x.view(x.size(0), -1).mean(1).view(*shape)
        std = x.view(x.size(0), -1).std(1).view(*shape)
        x = (x - mean) / (std + self.eps)

        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x
    
###########################################
# Generative architecture H
###########################################

class GANwritingGenerator(nn.Module):
    def __init__(self, imsize=(64,960), string_len=17):
        super(GANwritingGenerator, self).__init__()
        self.generator = Decoder()
        num_adain_params = self.generator.get_num_adain_params()
        self.content_encoder = ContentEncoder(num_adain_params, string_len=string_len)
        self.style_encoder = VGG19_BN()
        
        self.imsize = imsize
        self.input_feat_size = (imsize[0] // 16, imsize[1] // 16)
        assert imsize[0] % 16 == 0 and imsize[1] % 16 == 0
        
    def forward(self, x, t):
        Fs = self.style_encoder(x)
        Z = torch.randn_like(Fs) # noise
        Fs = Fs + Z
        Fc, fc = self.content_encoder(t)
        
        # Repeat characters in Fc to coarsely correspond to their intended locations
#         # Horizontally
#         Fc = Fc.transpose(1,2).repeat_interleave(self.imsize[0] // 16, dim=2)
#         # Vertically
#         Fc = Fc[:,:,None,:].repeat_interleave(self.imsize[0] // 16, dim=2)
        Fc = nn.functional.interpolate(Fc.transpose(1,2)[:,:,None,:], size=self.input_feat_size)
        
        F = torch.cat([Fs, Fc], dim=1)
        img = self.generator(F, fc)
        return img
    

###########################################
# Discriminator
###########################################

    
class Discriminator(nn.Module):
    def __init__(self, num_classes=1, imsize=(64,960)):
        super(Discriminator, self).__init__()
        # TODO: replace pooling with stride in residual blocks
        layers = [nn.Conv2d(1, 64, 3, padding=1), nn.LeakyReLU()]
        # Residual blocks
        layers += [BasicResidualBlock(64, 128), nn.AvgPool2d(2),
                   BasicResidualBlock(128, 128), nn.AvgPool2d(2),
                   BasicResidualBlock(128, 256), nn.AvgPool2d(2),
                   BasicResidualBlock(256, 256), nn.AvgPool2d(2),
                   BasicResidualBlock(256, 512), nn.AvgPool2d(2),
                   BasicResidualBlock(512, 512), nn.AvgPool2d(2)]
        # Classifier
        if num_classes == 1:
            # Binary classifier with single output
            layers += [nn.Conv2d(512, 1, (imsize[0]//64,imsize[1]//64)), nn.Sigmoid()] # Sigmoid for BCELoss()
        else:
            layers += [nn.Flatten(), MultiLayerPerceptron(512*(imsize[0]//64)*(imsize[1]//64), num_classes)] # for CrossEntropyLoss()
        
        self.model = nn.Sequential(*layers)
        self.num_classes = num_classes

    def forward(self, x):
        x = self.model(x)
        if self.num_classes == 1:
            # Binary classifier with single output
            x = x.view(-1) # squeeze
        return x
    
    
class BasicResidualBlock(nn.Module):
    def __init__(self, inplanes, planes, norm_layer=nn.BatchNorm2d):
        super(BasicResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, padding=1)
        self.bn1 = norm_layer(planes)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1)
        self.bn2 = norm_layer(planes)
        if inplanes != planes:
            self.conv3 = nn.Conv2d(inplanes, planes, 1)
        else:
            self.conv3 = None

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.conv3 is not None:
            identity = self.conv3(x)
        out += identity
        out = self.relu(out)
        return out
    
    
###########################################
# Wrod Recognizer Seq2Seq
###########################################  
# Reference: https://github.com/omni-us/research-seq2seq-HTR

class Seq2SeqEncoder(nn.Module):
    SUM_UP = True
    
    def __init__(self, hidden_size, imsize, n_rnn_layers=2):
        super(Seq2SeqEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.height = imsize[0]
        self.width = imsize[1]
        self.n_rnn_layers = n_rnn_layers

        self.vgg = VGG19_BN(in_channels=1)

        RNN = nn.GRU

        self.rnn = RNN(self.height//16*512, self.hidden_size, self.n_rnn_layers, dropout=0.5, bidirectional=True)
        if self.SUM_UP:
            self.enc_out_merge = lambda x: x[:,:,:x.shape[-1]//2] + x[:,:,x.shape[-1]//2:]

    def forward(self, in_data, in_data_len, hidden=None):
        out = self.vgg(in_data)
        # Reshape to (width, batch, height x channels)
        out = out.permute(3, 0, 2, 1) 
        out = out.reshape(out.shape[0], out.shape[1], out.shape[2] * out.shape[3])
        
        width = out.shape[0]
        src_len = in_data_len.numpy()*(width/self.width)
        src_len = src_len + 0.999 # in case of 0 length value from float to int
        src_len = src_len.astype('int')
        out = pack_padded_sequence(out, src_len.tolist(), batch_first=False)
        output, hidden = self.rnn(out, hidden)
        # output: t, b, f*2  hidden: 2, b, f
        output, output_len = pad_packed_sequence(output, batch_first=False)
        if self.SUM_UP:
            output = self.enc_out_merge(output)
        # output: t, b, f    hidden:  b, f
        odd_idx = [1, 3, 5, 7, 9, 11]
        hidden_idx = odd_idx[:self.n_rnn_layers]
        final_hidden = hidden[hidden_idx]
        return output, final_hidden # t, b, f*2    b, f*2
    
    
class locationAttention(nn.Module):
    ATTN_SMOOTH = False
    def __init__(self, hidden_size, decoder_layer):
        super(locationAttention, self).__init__()
        k = 128 # the filters of location attention
        r = 7 # window size of the kernel
        self.hidden_size = hidden_size
        self.decoder_layer = decoder_layer
        self.proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.tanh = nn.Tanh()
        self.hidden_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.encoder_output_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(hidden_size, 1)
        self.conv1d = nn.Conv1d(1, k, r, padding=3)
        self.prev_attn_proj = nn.Linear(k, self.hidden_size)
        self.softmax = nn.Softmax(dim=0)
        self.sigmoid = nn.Sigmoid()
        if self.ATTN_SMOOTH:
            self.sigma = self.attn_smoothing
        else:
            self.sigma = self.softmax

    def attn_smoothing(self, x):
        return self.sigmoid(x) / self.sigmoid(x).sum()

    # hidden:         layers, b, f
    # encoder_output: t, b, f
    # prev_attention: b, t
    def forward(self, hidden, encoder_output, enc_len, prev_attention):
        encoder_output = encoder_output.transpose(0, 1) # b, t, f
        attn_energy = self.score(hidden, encoder_output, prev_attention)

        attn_weight = torch.zeros(attn_energy.shape, device='cuda')
        for i, le in enumerate(enc_len):
            attn_weight[i, :le] = self.sigma(attn_energy[i, :le])
        return attn_weight.unsqueeze(2)

    # encoder_output: b, t, f
    def score(self, hidden, encoder_output, prev_attention):
        hidden = hidden.permute(1, 2, 0) # b, f, layers
        addMask = torch.FloatTensor([1/self.decoder_layer] * self.decoder_layer).view(1, self.decoder_layer, 1)
        addMask = torch.cat([addMask] * hidden.shape[0], dim=0)
        addMask = addMask.cuda() # b, layers, 1
        hidden = torch.bmm(hidden, addMask) # b, f, 1
        hidden = hidden.permute(0, 2, 1) # b, 1, f
        hidden_attn = self.hidden_proj(hidden) # b, 1, f

        prev_attention = prev_attention.unsqueeze(1) # b, 1, t
        conv_prev_attn = self.conv1d(prev_attention) # b, k, t
        conv_prev_attn = conv_prev_attn.permute(0, 2, 1) # b, t, k
        conv_prev_attn = self.prev_attn_proj(conv_prev_attn) # b, t, f

        encoder_output_attn = self.encoder_output_proj(encoder_output)
        res_attn = self.tanh(encoder_output_attn + hidden_attn + conv_prev_attn)
        out_attn = self.out(res_attn) # b, t, 1
        out_attn = out_attn.squeeze(2) # b, t
        return out_attn
    
    
class Seq2SeqDecoder(nn.Module):
    MULTINOMIAL = False
    def __init__(self, hidden_size, emb_size, vocab_size):
        super(Seq2SeqDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.emb_size = emb_size
        self.n_layers = 2
        self.embedding = nn.Embedding(vocab_size, self.emb_size)
        self.dropout = 0.5
        self.attention = locationAttention(self.hidden_size, self.n_layers)
        self.gru = nn.GRU(self.emb_size + self.hidden_size, self.hidden_size, self.n_layers, dropout=self.dropout)
        self.out = nn.Linear(self.hidden_size, vocab_size)

    def forward(self, in_char, hidden, encoder_output, src_len, prev_attn):
        width = encoder_output.shape[0]
        enc_len = src_len.numpy() * (width/src_len[0].item())
        enc_len = enc_len + 0.999
        enc_len = enc_len.astype('int')
        attn_weights = self.attention(hidden, encoder_output, enc_len, prev_attn)

        encoder_output_b = encoder_output.permute(1, 2, 0)
        context = torch.bmm(encoder_output_b, attn_weights)
        context = context.squeeze(2)

        if self.MULTINOMIAL and self.training:
            top1 = torch.multinomial(in_char, 1)
        else:
            top1 = in_char.topk(1)[1]
        embed_char = self.embedding(top1)
        embed_char = embed_char.squeeze(1)

        in_dec = torch.cat((embed_char, context), 1)
        in_dec = in_dec.unsqueeze(0)
        output, latest_hidden = self.gru(in_dec, hidden)
        output = output.squeeze(0)
        output = self.out(output)
        return output, latest_hidden, attn_weights.squeeze(2)
    
    
class Seq2Seq(nn.Module):
    def __init__(self, imsize=(64,960), max_len=17+1, emb_size=20, vocab_size=27):
        super(Seq2Seq, self).__init__()
        self.encoder = Seq2SeqEncoder(512, imsize)
        self.decoder = Seq2SeqDecoder(512, emb_size, vocab_size)
        self.max_len = max_len
        self.vocab_size = vocab_size

    def forward(self, src, tar, src_len, teacher_rate=0, train=True):
        tar = tar.transpose(1, 0) # time_s, batch, vocabulary
        batch_size = src.size(0)
        outputs = torch.zeros(self.max_len-1, batch_size, self.vocab_size, device='cuda')
        out_enc, hidden_enc = self.encoder(src, src_len)

        output = tar[0].data
        attns = []

        hidden = hidden_enc
        attn_weights = torch.zeros(out_enc.shape[1], out_enc.shape[0], device='cuda') # b, t

        for t in range(0, self.max_len-1): # max_len: groundtruth + <END>
            output, hidden, attn_weights = self.decoder(
                    output, hidden, out_enc, src_len, attn_weights)
            outputs[t] = output
            
            teacher_force_rate = random.random() < teacher_rate
            if train and teacher_force_rate:
                output = tar[t+1].data
            else:
                output = output.data
                
            attns.append(attn_weights.data.cpu())
        return outputs, attns