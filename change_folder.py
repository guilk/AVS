import os

if __name__ == '__main__':
    split_names = ['sixth', 'seventh']

    dst_root = '/data2/subset_V3C'
    for split_name in split_names:
        fw = open('../{}_gpu6.txt'.format(split_name), 'wb')
        with open('./{}_split.txt'.format(split_name), 'rb') as fr:
            lines = fr.readlines()
            for line in lines:
                line = line.rstrip('\r\n')
                folder_name = line.split('/')[-2]
                video_name = line.split('/')[-1]
                dst_path = os.path.join(dst_root, folder_name, video_name)
                fw.write(dst_path+'\n')
        fw.close()