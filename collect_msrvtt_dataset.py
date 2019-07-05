import os
import random


if __name__ == '__main__':

    root_path = '/mnt/sda/AVS/MSR-VTT/'
    videos = os.listdir(os.path.join(root_path, 'TrainValVideo'))
    fw = open('./msrvtt_trainval_data.txt', 'wb')
    for video_name in videos:
        video_path = os.path.join(root_path, 'TrainValVideo', video_name)
        # feat_path = os.path.join(root_path, 'features', video_name.replace('.avi','.npy'))
        # line = '{} {}\n'.format(video_path, feat_path)
        line = '{}\n'.format(video_path)
        fw.write(line)