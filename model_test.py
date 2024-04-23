import convlstm

import torch
import torch.nn as nn
from torch.nn import functional as F

import copy


class Predictor(nn.Module):
    def __init__(self, args):
        super(Predictor, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(3, 3), stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=1),
            nn.ELU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=1, padding=1),
            nn.ELU())
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=(3, 3), stride=1, padding=1, output_padding=0),
            nn.ELU(),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(3, 3), stride=2, padding=1, output_padding=1),
            nn.ELU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=1, output_padding=0),
            nn.ELU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=(3, 3), stride=2, padding=1, output_padding=1))

        if args.dataset == 'kth':
            self.decoder.add_module("last_activation", nn.Sigmoid())

        self.convlstm_num = 4
        self.convlstm_in_c = [128, 128, 128, 128]
        self.convlstm_out_c = [128, 128, 128, 128]
        self.convlstm_list = []
        for layer_i in range(self.convlstm_num):
            self.convlstm_list.append(convlstm.NPUnit(in_channels=self.convlstm_in_c[layer_i],
                                                      out_channels=self.convlstm_out_c[layer_i],
                                                      kernel_size=[3, 3]))
        self.convlstm_list = nn.ModuleList(self.convlstm_list)
        
        self.memory = Memory(args.memory_size, args.short_len, args.long_len)

        self.attention_size = 128
        self.attention_func = nn.Sequential(
            nn.AdaptiveAvgPool2d([1, 1]),
            nn.Flatten(),
            nn.Linear(256, 16),
            nn.ReLU(),
            nn.Linear(16, self.attention_size),
            nn.Sigmoid())

    def forward(self, short_x, long_x, out_len=16, phase=1):
    # def forward(self, short_x, long_x, out_len, phase):
        batch_size = short_x.size()[0]
        input_len= short_x.size()[1]

        # long-term motion context recall
        memory_x = long_x if phase == 1 else short_x
        # exit(0)
        memory_feature = self.memory(memory_x, phase)

        # motion context-aware video prediction
        h, c, out_pred = [], [], []
        for layer_i in range(self.convlstm_num):
            zero_state = torch.zeros(batch_size, self.convlstm_in_c[layer_i], memory_feature.size()[2], memory_feature.size()[3]).to(self.device)
            h.append(zero_state)
            c.append(zero_state)
        for seq_i in range(input_len+out_len-1):
            if seq_i < input_len:
                input_x = short_x[:, seq_i, :, :, :]
                input_x = self.encoder(input_x)
            else:
                input_x = self.encoder(out_pred[-1])

            for layer_i in range(self.convlstm_num):
                if layer_i == 0:
                    h[layer_i], c[layer_i] = self.convlstm_list[layer_i](input_x, h[layer_i], c[layer_i])
                else:
                    h[layer_i], c[layer_i] = self.convlstm_list[layer_i](h[layer_i-1], h[layer_i], c[layer_i])

            if seq_i >= input_len-1:
                attention = self.attention_func(torch.cat([c[-1], memory_feature], dim=1))
                attention = torch.reshape(attention, (-1, self.attention_size, 1, 1))
                memory_feature_att = memory_feature * attention
                out_pred.append(self.decoder(torch.cat([h[-1], memory_feature_att], dim=1)))

        out_pred = torch.stack(out_pred)
        out_pred = out_pred.transpose(dim0=0, dim1=1)
        out_pred = out_pred[:, -out_len:, :, :, :]
        return out_pred
    

class MotionEncoder(nn.Module):
    def __init__(self, length):
        super(MotionEncoder, self).__init__()

        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(3, 3), stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=1),
            nn.ELU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=1, padding=1),
            nn.ELU())
        
        self.spatial_conv01 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=2, padding=1),
            nn.ReLU()
        )

        self.spatial_conv12 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=2, padding=1),
            nn.ReLU()
        )

        self.length = length-1 # 差分序列

        self.temporal_conv001 = nn.Sequential(
            nn.Conv2d(in_channels=128*self.length, out_channels=128*(self.length//2), kernel_size=(1, 1), stride=1, padding=0),
            nn.ReLU()
        )

        self.temporal_conv012 = nn.Sequential(
            nn.Conv2d(in_channels=128*(self.length//2), out_channels=128, kernel_size=(1, 1), stride=1, padding=0),
            nn.ReLU()
        )

        self.temporal_conv101 = nn.Sequential(
            nn.Conv2d(in_channels=256*self.length, out_channels=256*(self.length//2), kernel_size=(1, 1), stride=1, padding=0),
            nn.ReLU()
        )

        self.temporal_conv112 = nn.Sequential(
            nn.Conv2d(in_channels=256*(self.length//2), out_channels=256, kernel_size=(1, 1), stride=1, padding=0),
            nn.ReLU()
        )

        self.temporal_conv201 = nn.Sequential(
            nn.Conv2d(in_channels=512*self.length, out_channels=512*(self.length//2), kernel_size=(1, 1), stride=1, padding=0),
            nn.ReLU()
        )

        self.temporal_conv212 = nn.Sequential(
            nn.Conv2d(in_channels=512*(self.length//2), out_channels=512, kernel_size=(1, 1), stride=1, padding=0),
            nn.ReLU()
        )

    def forward(self, x_raw):
        x = x_raw[:,1:,:,:,:] - x_raw[:,:-1,:,:,:]
        B, T, C, H, W = x.shape
        x = x.reshape(B*T,C,H,W) 

        x0 =  self.spatial_encoder(x)
        x1 = self.spatial_conv01(x0)
        x2 = self.spatial_conv12(x1)

        B0, C0, H0, W0 = x0.shape
        B1, C1, H1, W1 = x1.shape
        B2, C2, H2, W2 = x2.shape

        x0 = x0.view(B0//T, C0*T, H0, W0)
        x1 = x1.view(B1//T, C1*T, H1, W1)
        x2 = x2.view(B2//T, C2*T, H2, W2)
        
        res0 = self.temporal_conv012(self.temporal_conv001(x0))
        res1 = self.temporal_conv112(self.temporal_conv101(x1))
        res2 = self.temporal_conv212(self.temporal_conv201(x2))

        return res0, res1, res2 

class Memory(nn.Module):
    def __init__(self, memory_size, short_len, long_len):
        super(Memory, self).__init__()

        self.motion_matching_encoder = MotionEncoder(short_len)

        self.motion_context_encoder = MotionEncoder(long_len)
        
        self.embedder21 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=(3, 3), stride=2, padding=1, output_padding=1),
            nn.ReLU()
            )
        
        self.embedder11 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=(3, 3), stride=2, padding=1, output_padding=1),
            nn.ReLU()
        )
        
        self.embedder10 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=(3, 3), stride=2, padding=1, output_padding=1),
            nn.ReLU()
        )

        self.fusion = nn.Sequential(
            nn.ConvTranspose2d(in_channels=384, out_channels=128, kernel_size=(1, 1), stride=1, padding=0),
            nn.ReLU()
        )

        self.memory_shape2 = [memory_size, 512]
        self.memory_shape1 = [memory_size*2, 256]
        self.memory_shape0 = [memory_size*4, 128]

        self.memory_w0 = nn.init.normal_(torch.empty(self.memory_shape0), mean=0.0, std=1.0)
        self.memory_w0 = nn.Parameter(self.memory_w0, requires_grad=True)

        self.memory_w1 = nn.init.normal_(torch.empty(self.memory_shape1), mean=0.0, std=1.0)
        self.memory_w1 = nn.Parameter(self.memory_w1, requires_grad=True)

        self.memory_w2 = nn.init.normal_(torch.empty(self.memory_shape2), mean=0.0, std=1.0)
        self.memory_w2 = nn.Parameter(self.memory_w2, requires_grad=True)

    def forward(self, memory_x, phase):
        # memory_x = memory_x[:, 1:, :, :, :] - memory_x[:, :-1, :, :, :] # make difference frames

        # memory_x = memory_x.transpose(dim0=1, dim1=2) # make (N, C, T, H, W) for 3D Conv
        motion_encoder = self.motion_context_encoder if phase == 1 else self.motion_matching_encoder
        # print(memory_x.shape)
        memory_query0, memory_query1, memory_query2 = motion_encoder(memory_x)
        # memory_query = torch.squeeze(motion_encoder(memory_x), dim=2) # make (N, C, H, W)
        # print(memory_query0.shape, memory_query1.shape, memory_query2.shape)

        memory_query = memory_query2
        query_c, query_h, query_w = memory_query.size()[1], memory_query.size()[2], memory_query.size()[3]
        memory_query = memory_query.permute(0, 2, 3, 1) # make (N, H, W, C)
        memory_query = torch.reshape(memory_query, (-1, query_c)) # make (N*H*W, C)

        # memory addressing
        query_norm = F.normalize(memory_query, dim=1)
        memory_norm = F.normalize(self.memory_w2, dim=1)
        s = torch.mm(query_norm, memory_norm.transpose(dim0=0, dim1=1))
        addressing_vec = F.softmax(s, dim=1)
        memory_feature = torch.mm(addressing_vec, self.memory_w2)

        memory_feature = torch.reshape(memory_feature, (-1, query_h, query_w, query_c)) # make (N, H, W, C)
        memory_feature = memory_feature.permute(0, 3, 1, 2) # make (N, C, H, W) for 2D DeConv
        memory_feature2 = self.embedder21(memory_feature)

        #----------------------------------

        memory_query = memory_query1 + memory_feature2
        query_c, query_h, query_w = memory_query.size()[1], memory_query.size()[2], memory_query.size()[3]
        memory_query = memory_query.permute(0, 2, 3, 1) # make (N, H, W, C)
        memory_query = torch.reshape(memory_query, (-1, query_c)) # make (N*H*W, C)
        

        # memory addressing
        query_norm = F.normalize(memory_query, dim=1)
        memory_norm = F.normalize(self.memory_w1, dim=1)
        s = torch.mm(query_norm, memory_norm.transpose(dim0=0, dim1=1))
        addressing_vec = F.softmax(s, dim=1)
        memory_feature = torch.mm(addressing_vec, self.memory_w1)

        memory_feature = torch.reshape(memory_feature, (-1, query_h, query_w, query_c)) # make (N, H, W, C)
        memory_feature = memory_feature.permute(0, 3, 1, 2) # make (N, C, H, W) for 2D DeConv
        memory_feature1 = self.embedder10(memory_feature)

        #----------------------------------

        memory_query = memory_query0 + memory_feature1
        query_c, query_h, query_w = memory_query.size()[1], memory_query.size()[2], memory_query.size()[3]
        memory_query = memory_query.permute(0, 2, 3, 1) # make (N, H, W, C)
        memory_query = torch.reshape(memory_query, (-1, query_c)) # make (N*H*W, C)

        # memory addressing
        query_norm = F.normalize(memory_query, dim=1)
        memory_norm = F.normalize(self.memory_w0, dim=1)
        s = torch.mm(query_norm, memory_norm.transpose(dim0=0, dim1=1))
        addressing_vec = F.softmax(s, dim=1)
        memory_feature = torch.mm(addressing_vec, self.memory_w0)

        memory_feature = torch.reshape(memory_feature, (-1, query_h, query_w, query_c)) # make (N, H, W, C)
        memory_feature0 = memory_feature.permute(0, 3, 1, 2) # make (N, C, H, W) for 2D DeConv

        return self.fusion(torch.cat((memory_feature0, memory_feature1, self.embedder11(memory_feature2)), 1)) 
