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
    first_half = video_lst[:len(video_lst)/2]
    second_half = video_lst[len(video_lst)/2:]

    with open('./first_half.txt', 'wb') as fw:
        for video_path in first_half:
            fw.write(video_path+'\n')

    with open('./second_half.txt', 'wb') as fw:
        for video_path in second_half:
            fw.write(video_path+'\n')
    print '{},{}'.format(len(first_half), len(second_half))