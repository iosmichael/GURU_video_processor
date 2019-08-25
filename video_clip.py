import cv2
import numpy as np
import time
import argparse
import os

def check_files(dir):
    if not os.path.exists(os.path.join(dir, 'world.mp4')):
        print('Required: {} not found'.format(os.path.exists(dir, 'world.mp4')))
        return False
    if not os.path.exists(os.path.join(dir, 'world_pupil_data.npy')):
        print('Required: {} not found'.format(os.path.join(dir, 'world_pupil_data.npy')))
        return False
    print('Find all required files from {}'.format(dir))
    return True

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Clipping for Pupil Data and Video')
    parser.add_argument('--data_folder', default='data', type=str, help='pupil data directory for processing gaze information')
    parser.add_argument('--num', default=1, type=int, help='how many segments to clip to')

    args = parser.parse_args()
    if check_files(args.data_folder):
        gaze_pos = np.load(os.path.join(args.data_folder,'world_pupil_data.npy'), allow_pickle=True)
        cap = cv2.VideoCapture(os.path.join(args.data_folder,'world.mp4'))
        # Check if camera opened successfully
        if (cap.isOpened()== False): 
            print("Error opening video stream or file")

        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print('Frame length: {}'.format(length))
        print('Num of Gaze positions: {}'.format(gaze_pos.shape[0]))

        if length != gaze_pos.shape[0]:
            print('Error: Frame length does not match positions from Gaze Data')
            exit()

        # Pre-allocate number of split
        segment = length // args.num
        bucket = 1
        f_index = 0

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        FRAME_WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        FRAME_HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        FPS = int(cap.get(cv2.CAP_PROP_FPS))
        print('ORIGINAL FPS: {}'.format(FPS))
        print('IMG SHAPE ({},{})'.format(FRAME_WIDTH, FRAME_HEIGHT))
        writer = cv2.VideoWriter(os.path.join(args.data_folder, "world_bucket{}.mp4".format(bucket)), fourcc, FPS, (FRAME_WIDTH, FRAME_HEIGHT))
        data = []
        while(cap.isOpened()):
            # Capture frame-by-frame
            ret, frame = cap.read()
            if ret == True:
                if f_index >= bucket * segment:
                    print('change writer at bucket {}'.format(bucket))
                    np.save(os.path.join(args.data_folder, "world_pupil_data_bucket{}.npy".format(bucket)), data)
                    writer.release()
                    data = []
                    bucket += 1
                    writer = cv2.VideoWriter(os.path.join(args.data_folder, "world_bucket{}.mp4".format(bucket)), fourcc, FPS, (FRAME_WIDTH, FRAME_HEIGHT))

                # height, width, _ = frame.shape
                norm_x, norm_y = gaze_pos[f_index]['norm_pos']
                conf = gaze_pos[f_index]['confidence']
                # Display the resulting frame
                data.append({
                    'norm_pos':gaze_pos[f_index]['norm_pos'],
                    'confidence': gaze_pos[f_index]['confidence']
                    })
                writer.write(frame)
                f_index += 1
                # Press Q on keyboard to  exit
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    breakå
            else:
                print('End of frame, exiting the program')
                break
        # When everything done, release the video capture object
        cap.release()
        np.save(os.path.join(args.data_folder, "world_pupil_data_bucket{}.npy".format(bucket)), data)
        writer.release()
        # Closes all the frames
        cv2.destroyAllWindows()