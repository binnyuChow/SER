# Copyright (c) 2020 Anita Hu and Kevin Su
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import glob
import os

# from networks.spec_cnn import DistillableViT
# from networks.spec_cnnn import SER_AlexNet

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import datetime
import argparse
import random
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch.utils.data as data
import torchvision.transforms as transforms
from networks import *
from ravdess import RAVDESSDataset
from main_utils import train, validation
import warnings
# from transformers import  Wav2Vec2Model
# from  timm.models import
from torchvision.models import resnet50
from torchvision.models import efficientnet_b3
# from timm.models.swin_transformer_v2 import

warnings.filterwarnings('ignore')

# setting seed
seed = 1234
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)  # cpu
torch.cuda.manual_seed_all(seed)  # gpu
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

# define model input
def get_X(device, sample):
    # images = sample["images"].to(device)
    # images = images.permute(0, 2, 1, 3, 4)  # swap to be (N, C, D, H, W) 原始（N D C H W）
    mfcc = sample["mfcc"].to(device) # [16, 13, 212]
    # mfcc = mfcc.permute(0,2,1)
    spec = sample["spec"].to(device)
    # wav = sample["wav"].to(device)
    # wav = torch.squeeze(wav, dim=1)
    n = mfcc[0].size(0)
    # emotion = sample["emotion"].to(device)
    # return [images, mfcc, spec, wav], n
    return [ mfcc, spec ], n



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str, help='dataset directory', default='RAVDESS/preprocessed')
    parser.add_argument('--k_fold', type=int, help='k for k fold cross validation', default=5)
    parser.add_argument('--lr', type=float, help='learning rate', default=0.001)
    parser.add_argument('--batch_size', type=int, help='batch size', default=60)
    parser.add_argument('--num_workers', type=int, help='num workers', default=4)
    parser.add_argument('--epochs', type=int, help='train epochs', default=100)
    parser.add_argument('--checkpointdir', type=str, help='directory to save/read weights', default='checkpoints')
    parser.add_argument('--no_verbose', action='store_true', default=False, help='turn off verbose for training')
    parser.add_argument('--log_interval', type=int, help='interval for displaying training info if verbose', default=10)
    parser.add_argument('--no_save', action='store_true', default=False, help='set to not save model weights')
    parser.add_argument('--train', action='store_true', default=True, help='training')

    args = parser.parse_args()

    print("The configuration b3+cfn-sr of this run is:")
    print(args, end='\n\n')

    # Detect devices
    use_cuda = torch.cuda.is_available()  # check if GPU exists
    device = torch.device("cuda" if use_cuda else "cpu")  # use CPU or GPU


    # Data loading parameters
    params = {'batch_size': args.batch_size, 'shuffle': True, 'num_workers': args.num_workers, 'pin_memory': True} \
        if use_cuda else {'batch_size': args.batch_size, 'shuffle': True, 'num_workers': args.num_workers}

    # define data transform
    train_transform = {
        "image_transform": transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.4)),
            # transforms.RandomHorizontalFlip(),

            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
        "audio_transform": False
    }

    val_transform = {
        "image_transform": transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.4)),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
        "audio_transform": False
    }

    # loss function
    loss_func = torch.nn.CrossEntropyLoss()

    # top k categorical accuracy: k
    training_topk = (1,)
    val_topk = (1, 2, 4,)

    all_folder = sorted(list(glob.glob(os.path.join(args.datadir, "Actor*"))))
    s = int(len(all_folder) / args.k_fold)  # size of a fold
    top_scores = []
    top_ua_scores = []
# 5折交叉验算
    for i in range(args.k_fold):
        print("Fold " + str(i + 1))

        # define dataset
        if args.train:
            train_fold = all_folder[:i * s] + all_folder[i * s + s:]
            training_set = RAVDESSDataset(train_fold, transform=train_transform)
            training_loader = data.DataLoader(training_set, **params)
            print("Train fold: ")
            print([os.path.basename(act) for act in train_fold])

            # record training process
            current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            train_log_dir = os.path.join(args.checkpointdir, 'logs/fold{}_{}'.format(i + 1, current_time))
            writer = SummaryWriter(log_dir=train_log_dir)

        val_fold = all_folder[i * s: i * s + s]
        val_set = RAVDESSDataset(val_fold, transform=val_transform)
        val_loader = data.DataLoader(val_set, **params)

        print("val fold: ")
        print([os.path.basename(act) for act in val_fold])


        model_param = {}

        audio_model = MFCCNet(features_only=True)

        spec_model = efficientnet_b3(pretrained=True)


        model_param = {

            "audio": {
                "model": audio_model,
                "id": 0
            },
            "spec": {
                "model": spec_model,
                "id": 1
            },

        }
        multimodal_model = MSAFNet(model_param)
        print(get_n_params(multimodal_model))
        multimodal_model.to(device)

        # train / evaluate models
        if args.train:
            # Adam parameters
            num_parameters = multimodal_model.parameters()
            optimizer = torch.optim.Adam(num_parameters, lr=args.lr)
            # keep track of epoch test scores
            test = []
            ua =[]
            best_acc_1 = 0
            for epoch in range(args.epochs):
                # train, test model
                train_losses, train_scores = train(get_X, args.log_interval, multimodal_model, device, training_loader,
                                                   optimizer, loss_func, training_topk, epoch)
                epoch_test_loss, epoch_test_score,epoch_ua_score = validation(get_X, multimodal_model, device, loss_func, val_loader,
                                                               val_topk)

                if not args.no_save and epoch_test_score[0] > best_acc_1:
                    best_acc_1 = epoch_test_score[0]
                    torch.save(multimodal_model.state_dict(),
                               os.path.join(args.checkpointdir, 'fold_{}_msaf_ravdess_best.pth'.format(i + 1)))
                    print("Epoch {} model saved!".format(epoch + 1))

                # save results
                writer.add_scalar('Loss/train', np.mean(train_losses), epoch)
                writer.add_scalar('Loss/test', epoch_test_loss, epoch)
                for each_k, k_score in zip(training_topk, train_scores):
                    writer.add_scalar('Scores_top{}/train'.format(each_k), np.mean(k_score), epoch)
                for each_k, k_score in zip(val_topk, epoch_test_score):
                    writer.add_scalar('Scores_top{}/test'.format(each_k), np.mean(k_score), epoch)
                test.append(epoch_test_score)
                ua.append(epoch_ua_score)
                writer.flush()
            test = np.array(test)
            for j, each_k in enumerate(val_topk):
                max_idx = np.argmax(test[:, j])
                print('Best top {} test wa score {:.2f}% at epoch {}'.format(each_k, test[:, j][max_idx], max_idx + 1))
                # print('Best top {} test ua score {:.2f}% at epoch {}'.format(each_k, ua[:, j][max_idx_], max_idx_ + 1))
            max_idx_ = np.argmax(ua[:])
            print('Best test ua score {:.2f}% at epoch {}'.format(ua[max_idx_], max_idx_ + 1))
            top_scores.append(test[:, 0].max())
            top_ua_scores.append(ua[max_idx_])
        else:  # load and evaluate model
            model_path = os.path.join(args.checkpointdir, 'fold_{}_msaf_ravdess_best.pth'.format(i + 1))
            checkpoint = torch.load(model_path) if use_cuda else torch.load(model_path,
                                                                            map_location=torch.device('cpu'))
            multimodal_model.load_state_dict(checkpoint)
            epoch_test_loss, epoch_test_score = validation(get_X, multimodal_model, device, loss_func, val_loader,
                                                           val_topk)
            top_scores.append(epoch_test_score[0])

        print("WA Scores for each fold: ")
        print(top_scores)
        print("UA Scores for each fold: ")
        print(top_ua_scores)
        print("Averaged WA score for {} fold: {:.2f}%".format(args.k_fold, sum(top_scores) / len(top_scores)))
        print("Averaged UA score for {} fold: {:.2f}%".format(args.k_fold, sum(top_ua_scores) / len(top_ua_scores)))