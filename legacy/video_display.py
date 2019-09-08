import cv2
import numpy as np
import argparse
import os

def check_files(dir, bucket):
    if bucket:
        if not os.path.exists(os.path.join(dir, 'world_bucket{}.mp4'.format(bucket))):
            print('Required: {} not found'.format(os.path.exists(dir, 'world_bucket{}.mp4'.format(bucket))))
            return False
        if not os.path.exists(os.path.join(dir, 'world_pupil_data_bucket{}.npy'.format(bucket))):
            print('Required: {} not found'.format(os.path.join(dir, 'world_pupil_data_bucket{}.npy'.format(bucket))))
            return False
    else:
        if not os.path.exists(os.path.join(dir, 'world.mp4'.format(bucket))):
            print('Required: {} not found'.format(os.path.exists(dir, 'world.mp4'.format(bucket))))
            return False
        if not os.path.exists(os.path.join(dir, 'world_pupil_data.npy'.format(bucket))):
            print('Required: {} not found'.format(os.path.join(dir, 'world_pupil_data.npy'.format(bucket))))
            return False
    print('Find all required files from {}'.format(dir))
    return True

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process for Pupil Data')
    parser.add_argument('--folder', default='.', type=str, help='world video media folder')
    parser.add_argument('--bucket', default=None, type=int, help='bucket index')

    args = parser.parse_args()

    if args.bucket:
        cap = cv2.VideoCapture(os.path.join(args.folder, "world_bucket{}.mp4".format(args.bucket)))
        gaze_pos = np.load(os.path.join(args.folder, 'world_pupil_data_bucket{}.npy'.format(args.bucket)))
    else:
        cap = cv2.VideoCapture(os.path.join(args.folder, "world.mp4"))
        gaze_pos = np.load(os.path.join(args.folder, 'world_pupil_data.npy'))
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print('Frame Length: {}'.format(length))
    print('Num of Gaze Data: {}'.format(gaze_pos.shape[0]))
    if length != gaze_pos.shape[0]:
        print("Error, number of frames does not correspond to number of gaze positions, mis-aligned data")
        cap.release()
        cv2.destroyAllWindows()
        exit()

    FRAME_WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    FRAME_HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS = int(cap.get(cv2.CAP_PROP_FPS))
    print('ORIGINAL FPS: {}'.format(FPS))
    print('IMG SHAPE ({},{})'.format(FRAME_WIDTH, FRAME_HEIGHT))

    # Check if camera opened successfully
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")
        cap.release()
        cv2.destroyAllWindows()
        exit()        
     
    # Read until video is completed
    f_index = 0
    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            norm_x, norm_y = gaze_pos[f_index]['norm_pos']
            conf = gaze_pos[f_index]['confidence']
            topic = gaze_pos[f_index]['topic']
            if conf < 0.8:
                f_index += 1
                continue
            f_index += 1
            gaze_x, gaze_y = norm_x * FRAME_WIDTH, norm_y * FRAME_HEIGHT
            # Display the resulting frame
            if conf < 0.5:
                continue

            cv2.putText(frame, "Conf: {}".format(np.round(conf * 100, decimals=2)), (10,FRAME_HEIGHT-20), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
            cv2.putText(frame, "Topic: {}".format(topic), (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
            cv2.circle(frame, (int(gaze_x), int(gaze_y)), 12, (255,165,0), -1)
            cv2.imshow('Frame',frame)
            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
            # Break the loop
        else:
            break
    # When everything done, release the video capture object
    cap.release()
    # Closes all the frames
    cv2.destroyAllWindows()
