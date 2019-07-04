import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import sys
import argparse
import cv2
import time

# parser = argparse.ArgumentParser()
# parser.add_argument('-mode', type=str, help='rgb or flow')
# parser.add_argument('-load_model', type=str)
# parser.add_argument('-root', type=str)
# parser.add_argument('-gpu', type=str)
# parser.add_argument('-save_dir', type=str)
# parser.add_argument('--video_lst', type=str)

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=int, default=0, help='cuda device id')
parser.add_argument('--mode', type=str, default='rgb')
parser.add_argument('--video_lst', type=str)
parser.add_argument('--data_root', type=str, default='/mnt/sda/AVS')
parser.add_argument('--workers', type=int, default=1)
# parser.add_argument()

args = parser.parse_args()
# os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable

import torchvision
from torchvision import datasets, transforms
import videotransforms

import numpy as np

from pytorch_i3d import InceptionI3d

# from charades_dataset_full import Charades as Dataset
from V3C_dataset import V3C as Dataset



def load_rgb_frames(image_dir, vid, start, num):
    frames = []
    for i in range(start, start + num):
        img = cv2.imread(os.path.join(image_dir, vid, str(i).zfill(6) + '.jpg'))[:, :, [2, 1, 0]]
        w, h, c = img.shape
        if w < 226 or h < 226:
            d = 226. - min(w, h)
            sc = 1 + d / min(w, h)
            img = cv2.resize(img, dsize=(0, 0), fx=sc, fy=sc)
        img = (img / 255.) * 2 - 1
        frames.append(img)
    return np.asarray(frames, dtype=np.float32)

def process_frame(img):
    # img = img[:, :, [2, 1, 0]]
    w, h, c = img.shape
    if w < 226 or h < 226:
        d = 226. - min(w, h)
        sc = 1 + d / min(w, h)
        img = cv2.resize(img, dsize=(0, 0), fx=sc, fy=sc)
    img = (img / 255.) * 2 - 1
    return img

def crop_frames(imgs, crop_size):
    t, h, w, c = imgs.shape
    th, tw = crop_size, crop_size
    i = int(np.round((h - th) / 2.))
    j = int(np.round((w - tw) / 2.))
    imgs = imgs[:, i:i + th, j:j + tw, :]
    return imgs


if __name__ == '__main__':
    batch_size = 1
    buffer_size = 64
    crop_size = 224
    mode = args.mode
    device = torch.device('cuda:{}'.format(args.device))

    # setup dataset
    test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])
    video_lst = []
    with open(args.video_lst, 'rb') as fr:
        lines = fr.readlines()
        for line in lines:
            video_lst.append(line.rstrip('\r\n'))
    # imgs_root = '/mnt/sda/tmp'
    imgs_root = os.path.join(args.data_root, 'tmp')
    # feat_root = '/mnt/sda/features'
    # feat_root = os.path.join(args.data_root, 'features')
    video_feat_folder = os.path.join(args.data_root, 'features')
    if not os.path.exists(video_feat_folder):
        os.makedirs(video_feat_folder)

    # start_time = time.time()
    # dataset = Dataset(video_lst=video_lst, imgs_root=imgs_root,
    #                   mode=mode, transforms=test_transforms)
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1,
    #                                          pin_memory=True)
    # end_time = time.time()
    # print 'making dataloader: {}'.format(end_time - start_time)
    if mode == 'flow':
        i3d = InceptionI3d(400, in_channels=2)
    else:
        i3d = InceptionI3d(400, in_channels=3)

    model_path = './models/rgb_charades.pt'
    i3d.replace_logits(157)
    i3d.load_state_dict(torch.load(model_path))
    i3d.to(device)
    i3d.eval()

    for index, video_path in enumerate(video_lst):

        video_name = video_path.split('/')[-1]
        video_folder = video_path.split('/')[-2]
        # video_feat_folder = os.path.join(feat_root, video_folder)
        if not os.path.exists(video_feat_folder):
            os.makedirs(video_feat_folder)
        feat_name = video_name.split('.')[0]+'.npy'
        dst_path = os.path.join(video_feat_folder, feat_name)
        if os.path.exists(dst_path):
            continue
        # imgs_path = os.path.join(imgs_root, video_name)
        # decode_start = time.time()
        img_folder_path = os.path.join(imgs_root, video_name)
        if not os.path.exists(img_folder_path):
            os.makedirs(img_folder_path)
        cmd = 'ffmpeg -i {} {} -hide_banner -loglevel panic'.format(video_path,
                                                                    os.path.join(img_folder_path, '%06d.jpg'))
        os.system(cmd)
        # decode_end = time.time()
        # print 'decoding time : {}'.format(decode_end - decode_start)

        num_frames = len(os.listdir(os.path.join(imgs_root, video_name)))
        cur_frame = 0
        buffer_counter = 0
        features = []
        frames = []
        frame_names = [str(frame_index+1).zfill(6) + '.jpg' for frame_index in range(num_frames)]

        while len(frame_names) % buffer_size != 0:
            rotate_frames = frame_names[:buffer_size - len(frame_names) % buffer_size]
            frame_names += rotate_frames
        # if len(frame_names) % buffer_size != 0:
        #     rotate_frames = frame_names[:buffer_size - len(frame_names) % buffer_size]
        #     frame_names += rotate_frames
        # feat_start = time.time()
        print 'Process {}th of {} videos, {} frames, padded {} frames'.format(index, len(video_lst), num_frames,
                                                                              len(frame_names))
        dataset = Dataset(frame_names=frame_names, imgs_root=img_folder_path,
                          mode=mode, transforms=test_transforms, buffer_size=buffer_size)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=args.workers)
        for imgs in dataloader:
            inputs = imgs.to(device)
            # print inputs.size()
            # print inputs.size()
            buffer_feats = i3d.extract_features(inputs)
            # print buffer_feats.size()
            buffer_feats = buffer_feats.squeeze(0).permute(1, 2, 3, 0).data.cpu().numpy()
            features.append(buffer_feats)
        if len(features) == 0:
            cmd = 'rm -rf {}'.format(img_folder_path)
            os.system(cmd)
            continue
        features = np.concatenate(features, axis=0)
        # feat_end = time.time()
        # print 'extracting features : {}'.format(feat_end - feat_start)
        # print features.shape
        cmd = 'rm -rf {}'.format(img_folder_path)
        os.system(cmd)
        ave_feat = np.mean(features, axis=0)
        np.save(dst_path, ave_feat)
        # assert False
        #     # if os.path.exists(os.path.join(save_dir, name[0]+'.npy')):
        #     #     continue
        # for frame_name in frame_names:
        #     frame = cv2.imread(os.path.join(imgs_root, video_name, frame_name))[:, :, [2, 1, 0]]
        #     frame = process_frame(frame)
        #     if buffer_counter < buffer_size:
        #         frames.append(frame)
        #         buffer_counter += 1
        #     else:
        #         imgs = np.asarray(frames, dtype=np.float32)
        #         imgs = crop_frames(imgs, crop_size)
        #         inputs = torch.from_numpy(imgs.transpose([3, 0, 1, 2])).to(device)
        #         inputs = inputs.unsqueeze(0)
        #         buffer_feats = i3d.extract_features(inputs)
        #         buffer_feats = buffer_feats.squeeze(0).permute(1, 2, 3, 0).data.cpu().numpy()
        #         features.append(buffer_feats)
        #
        #         frames = []
        #         buffer_counter = 0
        #         frames.append(frame)
        #         buffer_counter += 1
        # if len(frames) != 0:
        #     # print 'remaining frames: {}'.format(len(frames))
        #     imgs = np.asarray(frames, dtype=np.float32)
        #     imgs = crop_frames(imgs, crop_size)
        #     inputs = torch.from_numpy(imgs.transpose([3, 0, 1, 2])).to(device)
        #     inputs = inputs.unsqueeze(0)
        #     buffer_feats = i3d.extract_features(inputs)
        #     buffer_feats = buffer_feats.squeeze(0).permute(1, 2, 3, 0).data.cpu().numpy()
        #     features.append(buffer_feats)
        # features = np.concatenate(features, axis=0)
        # # feat_end = time.time()
        # # print len(frame_names),features.shape
        # cmd = 'rm -rf {}'.format(img_folder_path)
        # os.system(cmd)
        # ave_feat = np.mean(features, axis=0)
        # np.save('tmp.npy', ave_feat)
        # print ave_feat.shape
        # # print 'extracting features : {}'.format(feat_end - feat_start)



    # for video_path in video_lst:
    #     video_name = video_path.split('/')[-1]
    #     # imgs_path = os.path.join(imgs_root, video_name)
    #     # decode_start = time.time()
    #     img_folder_path = os.path.join(imgs_root, video_name)
    #     if not os.path.exists(img_folder_path):
    #         os.makedirs(img_folder_path)
    #     cmd = 'ffmpeg -i {} {} -hide_banner -loglevel panic'.format(video_path,
    #                                                                 os.path.join(img_folder_path, '%06d.jpg'))
    #     os.system(cmd)
    #     # decode_end = time.time()
    #     # print 'decoding time : {}'.format(decode_end - decode_start)
    #
    #     feat_start = time.time()
    #     num_frames = len(os.listdir(os.path.join(imgs_root, video_name)))
    #     cur_frame = 0
    #     buffer_counter = 0
    #     features = []
    #     frames = []
    #     frame_names = [str(frame_index+1).zfill(6) + '.jpg' for frame_index in range(num_frames)]
    #     if len(frame_names) % buffer_size != 0:
    #         rotate_frames = frame_names[:buffer_size - len(frame_names) % buffer_size]
    #         frame_names += rotate_frames
    #
    #     for frame_name in frame_names:
    #         frame = cv2.imread(os.path.join(imgs_root, video_name, frame_name))[:, :, [2, 1, 0]]
    #         frame = process_frame(frame)
    #         if buffer_counter < buffer_size:
    #             frames.append(frame)
    #             buffer_counter += 1
    #         else:
    #             imgs = np.asarray(frames, dtype=np.float32)
    #             imgs = crop_frames(imgs, crop_size)
    #             inputs = torch.from_numpy(imgs.transpose([3, 0, 1, 2])).to(device)
    #             inputs = inputs.unsqueeze(0)
    #             buffer_feats = i3d.extract_features(inputs)
    #             buffer_feats = buffer_feats.squeeze(0).permute(1, 2, 3, 0).data.cpu().numpy()
    #             features.append(buffer_feats)
    #
    #             frames = []
    #             buffer_counter = 0
    #             frames.append(frame)
    #             buffer_counter += 1
    #     if len(frames) != 0:
    #         # print 'remaining frames: {}'.format(len(frames))
    #         imgs = np.asarray(frames, dtype=np.float32)
    #         imgs = crop_frames(imgs, crop_size)
    #         inputs = torch.from_numpy(imgs.transpose([3, 0, 1, 2])).to(device)
    #         inputs = inputs.unsqueeze(0)
    #         buffer_feats = i3d.extract_features(inputs)
    #         buffer_feats = buffer_feats.squeeze(0).permute(1, 2, 3, 0).data.cpu().numpy()
    #         features.append(buffer_feats)
    #     features = np.concatenate(features, axis=0)
    #     # feat_end = time.time()
    #     # print len(frame_names),features.shape
    #     cmd = 'rm -rf {}'.format(img_folder_path)
    #     os.system(cmd)
    #     ave_feat = np.mean(features, axis=0)
    #     np.save('tmp.npy', ave_feat)
    #     print ave_feat.shape
    #     # print 'extracting features : {}'.format(feat_end - feat_start)

        # for frame_index in range(num_frames):
        #     frame = cv2.imread(os.path.join(imgs_root, video_name,
        #                                   str(frame_index+1).zfill(6) + '.jpg'))[:, :, [2, 1, 0]]
        #     frame = process_frame(frame)
        #     if buffer_counter < buffer_size:
        #         frames.append(frame)
        #         buffer_counter += 1
        #     else:
        #         imgs = np.asarray(frames, dtype=np.float32)
        #         imgs = crop_frames(imgs, crop_size)
        #         inputs = torch.from_numpy(imgs.transpose([3, 0, 1, 2])).to(device)
        #         inputs = inputs.unsqueeze(0)
        #         buffer_feats = i3d.extract_features(inputs)
        #         buffer_feats = buffer_feats.squeeze(0).permute(1, 2, 3, 0).data.cpu().numpy()
        #         features.append(buffer_feats)
        #
        #         frames = []
        #         buffer_counter = 0
        #         frames.append(frame)
        #         buffer_counter += 1
        # if len(frames) != 0:
        #     imgs = np.asarray(frames, dtype=np.float32)
        #     imgs = crop_frames(imgs, crop_size)
        #     inputs = torch.from_numpy(imgs.transpose([3, 0, 1, 2])).to(device)
        #     inputs = inputs.unsqueeze(0)
        #     buffer_feats = i3d.extract_features(inputs)
        #     buffer_feats = buffer_feats.squeeze(0).permute(1, 2, 3, 0).data.cpu().numpy()
        #     features.append(buffer_feats)
        # features = np.concatenate(features, axis=0)
        # cmd = 'rm -rf {}'.format(img_folder_path)
        # os.system(cmd)


        # buffer_size = 128
        # second_feats = []
        # for video_path in video_lst:
        #     video_name = video_path.split('/')[-1]
        #     # imgs_path = os.path.join(imgs_root, video_name)
        #     img_folder_path = os.path.join(imgs_root, video_name)
        #     if not os.path.exists(img_folder_path):
        #         os.makedirs(img_folder_path)
        #     cmd = 'ffmpeg -i {} {} -hide_banner -loglevel panic'.format(video_path,
        #                                                                 os.path.join(img_folder_path, '%06d.jpg'))
        #     os.system(cmd)
        #     num_frames = len(os.listdir(os.path.join(imgs_root, video_name)))
        #     cur_frame = 0
        #     buffer_counter = 0
        #     features = []
        #     frames = []
        #     for frame_index in range(num_frames):
        #         frame = cv2.imread(os.path.join(imgs_root, video_name,
        #                                         str(frame_index + 1).zfill(6) + '.jpg'))[:, :, [2, 1, 0]]
        #         frame = process_frame(frame)
        #         if buffer_counter < buffer_size:
        #             frames.append(frame)
        #             buffer_counter += 1
        #         else:
        #             imgs = np.asarray(frames, dtype=np.float32)
        #             imgs = crop_frames(imgs, crop_size)
        #             inputs = torch.from_numpy(imgs.transpose([3, 0, 1, 2])).to(device)
        #             inputs = inputs.unsqueeze(0)
        #             buffer_feats = i3d.extract_features(inputs)
        #             print buffer_feats.size()
        #             buffer_feats = buffer_feats.squeeze(0).permute(1, 2, 3, 0).data.cpu().numpy()
        #             # second_feats = buffer_feats[0]
        #             features.append(buffer_feats)
        #
        #             frames = []
        #             buffer_counter = 0
        #             frames.append(frame)
        #             buffer_counter += 1
        #     if len(frames) != 0:
        #         print 'remaining frames: {}'.format(len(frames))
        #         imgs = np.asarray(frames, dtype=np.float32)
        #         imgs = crop_frames(imgs, crop_size)
        #         inputs = torch.from_numpy(imgs.transpose([3, 0, 1, 2])).to(device)
        #         inputs = inputs.unsqueeze(0)
        #         buffer_feats = i3d.extract_features(inputs)
        #         print buffer_feats.size()
        #         buffer_feats = buffer_feats.squeeze(0).permute(1, 2, 3, 0).data.cpu().numpy()
        #         features.append(buffer_feats)
        #     features = np.concatenate(features, axis=0)
        #     print features.shape
        #     second_feats = features[0]
        #     cmd = 'rm -rf {}'.format(img_folder_path)
        #     os.system(cmd)
        #     print np.amax(first_feats - second_feats)

            # for video_path in video_lst:
    #     # load video
    #     try:
    #         vcap = cv2.VideoCapture(video_path)
    #         if not vcap.isOpened():
    #             raise Exception("cannot open %s" % video_path)
    #     except Exception as e:
    #         raise e
    #     frame_count = vcap.get(cv2.CAP_PROP_FRAME_COUNT)
    #     video_name = video_path.split('/')[-1]
    #     cur_frame = 0
    #     frames = []
    #     buffer_counter = 0
    #     features = []
    #     while cur_frame < frame_count:
    #         suc, frame = vcap.read()
    #         if not suc:
    #             cur_frame += 1
    #             print 'Fail to load {}th frame of {}'.format(cur_frame, video_name)
    #             continue
    #         frame = process_frame(frame)
    #
    #         if buffer_counter < buffer_size:
    #             frames.append(frame)
    #             buffer_counter += 1
    #         else:
    #             imgs = np.asarray(frames, dtype=np.float32)
    #             imgs = crop_frames(imgs, crop_size)
    #             inputs = torch.from_numpy(imgs.transpose([3, 0, 1, 2])).to(device)
    #             inputs = inputs.unsqueeze(0)
    #             # inputs = torch.FloatTensor(imgs).to(device)
    #             # print inputs.size()
    #             buffer_feats = i3d.extract_features(inputs)
    #             buffer_feats = buffer_feats.squeeze(0).permute(1, 2, 3, 0).data.cpu().numpy()
    #             features.append(buffer_feats)
    #
    #             frames = []
    #             buffer_counter = 0
    #
    #         cur_frame += 1
    #     if len(frames) != 0:
    #         imgs = np.asarray(frames, dtype=np.float32)
    #         imgs = crop_frames(imgs, crop_size)
    #         # inputs = torch.FloatTensor(imgs).to(device)
    #         inputs = torch.from_numpy(imgs.transpose([3, 0, 1, 2])).to(device)
    #         inputs = inputs.unsqueeze(0)
    #         # print inputs.size()
    #         buffer_feats = i3d.extract_features(inputs)
    #         print buffer_feats.size()
    #         buffer_feats = buffer_feats.squeeze(0).permute(1, 2, 3, 0).data.cpu().numpy()
    #         features.append(buffer_feats)
    #     features = np.concatenate(features, axis=0)
    #     print frame_count, features.shape


    # for index, imgs in enumerate(dataloader):
    #     # print imgs.size()
    #     inputs = imgs.to(device)
    #     features = i3d.extract_features(inputs)
    #     features = features.squeeze(0).permute(1, 2, 3, 0).data.cpu().numpy()
    #     second_feats = features[0]
    #     print features.shape
    #     print np.amax(first_feats-second_feats)

    #     # assert False
    #     # print imgs.size()
    # # assert False
    # for phase in ['train', 'val']:
    #     i3d.train(False)  # Set model to evaluate mode
    #
    #     tot_loss = 0.0
    #     tot_loc_loss = 0.0
    #     tot_cls_loss = 0.0
    #
    #     # Iterate over data.
    #     for data in dataloaders[phase]:
    #         # get the inputs
    #         inputs, labels, name = data
    #         if os.path.exists(os.path.join(save_dir, name[0] + '.npy')):
    #             continue
    #
    #         b, c, t, h, w = inputs.shape
    #         if t > 1600:
    #             features = []
    #             for start in range(1, t - 56, 1600):
    #                 end = min(t - 1, start + 1600 + 56)
    #                 start = max(1, start - 48)
    #                 ip = Variable(torch.from_numpy(inputs.numpy()[:, :, start:end]).cuda(), volatile=True)
    #                 features.append(i3d.extract_features(ip).squeeze(0).permute(1, 2, 3, 0).data.cpu().numpy())
    #             np.save(os.path.join(save_dir, name[0]), np.concatenate(features, axis=0))
    #         else:
    #             # wrap them in Variable
    #             inputs = Variable(inputs.cuda(), volatile=True)
    #             features = i3d.extract_features(inputs)
    #             np.save(os.path.join(save_dir, name[0]), features.squeeze(0).permute(1, 2, 3, 0).data.cpu().numpy())