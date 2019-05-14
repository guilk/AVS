import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import sys
import argparse

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


if __name__ == '__main__':
    batch_size = 1
    mode = args.mode
    device = torch.device('cuda:{}'.format(args.device))

    # setup dataset
    test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])
    video_lst = []
    with open(args.video_lst, 'rb') as fr:
        lines = fr.readlines()
        for line in lines:
            video_lst.append(line.rstrip('\r\n'))
    imgs_root = '/mnt/sda/tmp'
    dataset = Dataset(video_lst=video_lst, imgs_root=imgs_root,
                      mode=mode, transforms=test_transforms)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1,
                                             pin_memory=True)

    if mode == 'flow':
        i3d = InceptionI3d(400, in_channels=2)
    else:
        i3d = InceptionI3d(400, in_channels=3)

    model_path = './models/rgb_charades.pt'
    i3d.replace_logits(157)
    i3d.load_state_dict(torch.load(model_path))
    i3d.to(device)
    i3d.eval()

    for index, imgs in enumerate(dataloader):
        print imgs.size()
        inputs = imgs.to(device)
        features = i3d.extract_features(inputs)
        print features.size()
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
