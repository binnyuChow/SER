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

import numpy as np
import glob
import os
import argparse
from tqdm import tqdm
import librosa
from video_utils import VideoProcessor
import pylab
from torchvision import transforms
from PIL import Image
import librosa.display
# 对原始视频提取音频及图片
def preprocess_video(dataset_folder):
    output_folder = os.path.join(dataset_folder, "preprocessed")
    landmark_folder = os.path.join(dataset_folder, "landmarks")

    for each_actor in glob.glob(os.path.join(dataset_folder, "Actor*")):
        actor_name = os.path.basename(each_actor)
        output_actor_folder = os.path.join(output_folder, actor_name)

        if not os.path.exists(output_actor_folder):
            os.makedirs(output_actor_folder)

        for each_video in glob.glob(os.path.join(each_actor, "*.mp4")):
            name = each_video.split("/")[-1].split("-")[0]
            if int(name) == 2:
                continue
            video_name = os.path.basename(each_video)
            landmark_path = os.path.join(landmark_folder, video_name[:-4] + '.csv')
            frames_folder = os.path.join(output_actor_folder, video_name)
            if not os.path.exists(frames_folder):
                os.mkdir(frames_folder)
            # video_processor = VideoProcessor(video_path=each_video, landmark_path=landmark_path,
            #                                  output_folder=frames_folder, extract_audio=True )
            video_processor = VideoProcessor(video_path=each_video, output_folder=frames_folder
                                         ,extract_audio=True )
            video_processor.preprocess(seq_len=30, target_resolution=(224, 224))

# 获取音频的路径
def get_audio_paths(path):
    audio_files = []
    actors = os.listdir(path)
    for a in actors:
        path_to_folders = os.path.join(path, a)
        folders = os.listdir(path_to_folders)
        audio_files += [os.path.join(path_to_folders, p) for p in folders]
    return audio_files

# 提取mfcc特征
def mfcc_features(dataset_folder):
    path = os.path.join(dataset_folder, "preprocessed")
    files = get_audio_paths(path)
    for f in tqdm(files):
        X, sample_rate = librosa.load(os.path.join(f, 'audios/audio.wav'),
                                      duration=3.00, sr=16000, offset=0.5)
        sample_rate = np.array(sample_rate)
        # mfccs = librosa.feature.mfcc(y=X, sr=sample_rate,  n_mfcc=40 )
        mfccs = librosa.feature.mfcc(y=X, sr=sample_rate,  n_mfcc=13)

        desired_shape = (13, 94)
        # desired_shape = (40, 241)
        n_mfcc, n_frames = mfccs.shape

        # 我们先创建一个desired_shape大小的零矩阵，并将mfcc_matrix插入到其中心
        padded_mfcc_matrix = np.zeros(desired_shape)
        start_row = (desired_shape[0] - n_mfcc) // 2
        start_col = (desired_shape[1] - n_frames) // 2
        padded_mfcc_matrix[start_row:start_row + n_mfcc, start_col:start_col + n_frames] = mfccs
        np.save(os.path.join(f, 'audios/featuresMFCC.npy'), padded_mfcc_matrix)


def melspec_features(dataset_folder):
    path = os.path.join(dataset_folder, "preprocessed")
    files = get_audio_paths(path)
    crop = transforms.Compose([
                # transforms.CenterCrop(288),
                # transforms.ToTensor(),
                transforms.Resize(224),
                # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
    pylab.axis('off') # no axis
    pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[]) # Remove the white edge
    for f in tqdm(files):
        X, sample_rate = librosa.load(os.path.join(f, 'audios/audio.wav'),
                                      duration=3.00, sr=16000, offset=0.5)


        S = librosa.feature.melspectrogram(y=X, sr=sample_rate, n_fft=800, hop_length=400)
        librosa.display.specshow(librosa.power_to_db(S, ref=np.max))

        logmelspec = librosa.power_to_db(S, ref=np.max)
        pylab.savefig(os.path.join(f, 'audios/spec.jpg'), bbox_inches=None, pad_inches=0)
        img = Image.open(os.path.join(f, 'audios/spec.jpg'))
        img = crop(img)
        img.save(os.path.join(f, 'audios/spec.jpg'))
        pylab.close()

        np.save(os.path.join(f, 'audios/featuresSPEC.npy'), logmelspec)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str, help='dataset directory', default='RAVDESS')
    args = parser.parse_args()

    # print("Processing videos...")
    # preprocess_video(args.datadir)

    print("Generating MFCC features...")
    mfcc_features(args.datadir)

    print("Generating SPEC features...")
    melspec_features(args.datadir)

    # print("

    print("Preprocessed dataset located in ", os.path.join(args.datadir, 'preprocessed'))
