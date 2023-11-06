
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

import torch.nn.functional as F
import sys
from modules.transformer_timm import AttentionBlock, Attention

# from transformers import  Wav2Vec2Model
sys.path.append('..')

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class MSAFNet(nn.Module):
    def __init__(self, model_param):
        super(MSAFNet, self).__init__()

        self.fc = nn.Linear(2000, 8)
        self.fc_ = nn.Linear(1000, 8)


        # mfcc
        self.lstm_mfcc = nn.LSTM(input_size=11, hidden_size=64, num_layers=2, batch_first=True, dropout=0.5,
                               bidirectional=True)
        self.mfcc_dropout = nn.Dropout(p=0.5)
        self.mfcc_linear = nn.Linear(8192, 1000)

        # spec
        # self.specnet = SER_AlexNet(num_classes=8, in_ch=3)
        self.spec_dropout = nn.Dropout(p=0.5)
        self.spec_linear = nn.Linear(2048, 128)
        # self.alexnet_model = SER_AlexNet(num_classes=8, in_ch=3, pretrained=False)

        # video
        self.video_dropout = nn.Dropout(p=0.5)
        self.video_linear = nn.Linear(15872, 128)

        #wav
        self.pool = torch.nn.AdaptiveMaxPool1d(1000)

        self.wav_mfcc_linear = nn.Linear(704, 1000)
        self.wav_dropout = nn.Dropout(p=0.5)
        self.wav_linear = nn.Linear(1024, 128)


        #mfcc_video
        self.mfcc_video_dropout = nn.Dropout(p=0.3)
        self.mfcc_video_linear = nn.Linear(212, 128)


        self.attn_layer1 = AttentionBlock(in_dim_k=1000, in_dim_q=1000, out_dim=1000, num_heads=1)
        # self.attn_layer1 = AttentionBlock(in_dim_k=1000, in_dim_q=1000, out_dim=1000, num_heads=1)
        if "video" in model_param:
            video_model = model_param["video"]["model"]  # 相当于  video_model = self.resnet50(audio_spec)

            self.video_model = video_model
            # self.video_model_blocks = self.make_blocks(video_model, self.msaf_locations["video"])
            self.video_id = model_param["video"]["id"]

        if "audio" in model_param:
            audio_model = model_param["audio"]["model"]

            # self.audio_model = audio_model
            self.audio_model = audio_model

            self.audio_id = model_param["audio"]["id"]


        if "spec" in model_param:
            spec_model = model_param["spec"]["model"]

            self.spec_model = spec_model
            # self.emotion = model_param["emotion"]["id"]
            self.spec_id = model_param["spec"]["id"]


    def forward(self, x):

        if hasattr(self, "audio_id"):
            mfcc = F.normalize(x[self.audio_id], p=2, dim=1)#[16,13,212]
            mfcc = mfcc.to(torch.float32)
            # mfcc = mfcc.permute(0,2,1)
            # mfcc_= torch.flip(mfcc,dims=[0,1])
            # mfcc, _ = self.gru_mfcc(mfcc)  # [16,# 212, 128]
            mfcc = self.audio_model(mfcc)  # [16,# 212, 1 28]
            mfcc,_ = self.lstm_mfcc(mfcc)  # [16,# 212, 128]
            mfcc = self.mfcc_dropout(mfcc)# ([16, 108544])
            mfcc = torch.flatten(mfcc,1) # ([16, 108544])
            mfcc = self.mfcc_linear(mfcc)
            mfcc = F.relu(mfcc, inplace=False)  # [batch, 128]
            mfcc = mfcc.unsqueeze(-1)
            mfcc = mfcc.permute(0,2,1)
            x[self.audio_id] = mfcc

        if hasattr(self, "spec_id"):
            spec = self.spec_model(x[self.spec_id]) # [16, 256, 6, 6]
            spec = spec.unsqueeze(-1)
            spec = spec.permute(0,2,1)
            # specx =spec.unsqueeze(2)
            x[self.spec_id] = spec

        x[self.audio_id] = self.attn_layer1(x[self.audio_id],x[self.spec_id])
        x[self.spec_id] = self.attn_layer1(x[self.spec_id],x[self.audio_id])
        x[self.audio_id] = x[self.audio_id].squeeze()
        x[self.spec_id] = x[self.spec_id].squeeze()
        a = self.pool(x[self.audio_id])
        b = self.pool(x[self.spec_id])

        res = torch.cat((a,b), dim=1)
        res = self.fc(res)
        # res = self.fc_(res+spec_)
        # print(res.shape)
        # res =   a + a_
        return res

