import os
import random


if __name__ == '__main__':
    root_path = '/mnt/sda/raw_data/tv2019/V3C1/V3C1.webm.videos.shots'
    subfolders = os.listdir(root_path)

    video_lst = []

    # 1082657 videos
    counter = 0
    for subfolder in subfolders:
        subfolder_path = os.path.join(root_path, subfolder)
        if not os.path.isdir(subfolder_path):
            continue
        videos = os.listdir(subfolder_path)
        videos = [video for video in videos if (not video.startswith('._')) and video.endswith('.webm')]
        for video_name in videos:
            video_path = os.path.join(subfolder_path, video_name)
            video_lst.append(video_path)
            counter += 1
            print video_path
    print 'The number of videos is {}'.format(counter)

    random.shuffle(video_lst)
    num_splits = 5
    seq_len = len(video_lst)/num_splits
    split_names = ['first', 'second', 'third', 'forth', 'fifth']

    for index in range(num_splits):
        data_list = video_lst[index*seq_len:(index+1)*seq_len]
        with open('./{}_split.txt'.format(split_names[index]), 'wb') as fw:
            for video_path in data_list:
                fw.write(video_path+'\n')
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