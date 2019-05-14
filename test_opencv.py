import cv2




if __name__ == '__main__':
    videofile = '../../shot03161_14.webm'
    try:
        vcap = cv2.VideoCapture(videofile)
        if not vcap.isOpened():
            raise Exception("cannot open %s" % videofile)
    except Exception as e:
        raise e

    frame_count = vcap.get(cv2.CAP_PROP_FRAME_COUNT)



    # vcap = cv2.VideoCapture(videofile)
