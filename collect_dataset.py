import os
import random


if __name__ == '__main__':
    root_path = '/mnt/sda/raw_data/tv2019/V3C1/V3C1.webm.videos.shots'
    subfolders = os.listdir(root_path)

    feat_root = '/mnt/sda/features'

    # video_lst = []
    #
    # # 1082657 videos
    # counter = 0
    # for subfolder in subfolders:
    #     subfolder_path = os.path.join(root_path, subfolder)
    #     if not os.path.isdir(subfolder_path):
    #         continue
    #     videos = os.listdir(subfolder_path)
    #     videos = [video for video in videos if (not video.startswith('._')) and video.endswith('.webm')]
    #     for video_name in videos:
    #
    #         video_feat_folder = os.path.join(feat_root, subfolder)
    #         feat_name = video_name.split('.')[0] + '.npy'
    #         if os.path.exists(feat_name):
    #             continue
    #         dst_path = os.path.join(video_feat_folder, feat_name)
    #         video_path = os.path.join(subfolder_path, video_name)
    #         video_lst.append(video_path)
    #         counter += 1
    #         print video_path
    # print 'The number of videos is {}'.format(counter)
    #
    # random.shuffle(video_lst)
    # num_splits = 7
    # seq_len = len(video_lst)/num_splits
    # split_names = ['first', 'second', 'third', 'forth', 'fifth', 'sixth', 'seventh']
    #
    # for index in range(num_splits):
    #     data_list = video_lst[index*seq_len:(index+1)*seq_len]
    #     with open('./{}_split.txt'.format(split_names[index]), 'wb') as fw:
    #         for video_path in data_list:
    #             fw.write(video_path+'\n')


    dst_root = '../subset_V3C'
    if not os.path.exists(dst_root):
        os.makedirs(dst_root)

    splits = ['sixth', 'seventh']
    for split_name in splits:
        with open('./{}_split.txt'.format(split_name), 'rb') as fr:
            lines = fr.readlines()
            for line in lines:
                src_path = line.rstrip('\r\n')
                subfolder = src_path.split('/')[-2]
                video_name = src_path.split('/')[-1]
                folder_path = os.path.join(dst_root, subfolder)
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)
                dst_path = os.path.join(folder_path, video_name)
                if os.path.exists(dst_path):
                    continue
                cmd = 'scp {} {}'.format(src_path, dst_path)
                print cmd
                os.system(cmd)
    # first_half = video_lst[:len(video_lst)/3]
    # second_half = video_lst[len(video_lst)/3:2*len(video_lst)/3]
    #
    # with open('./first_half.txt', 'wb') as fw:
    #     for video_path in first_half:
    #         fw.write(video_path+'\n')
    #
    # with open('./second_half.txt', 'wb') as fw:
    #     for video_path in second_half:
    #         fw.write(video_path+'\n')
    # print '{},{}'.format(len(first_half), len(second_half))