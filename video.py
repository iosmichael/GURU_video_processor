import cv2
import numpy as np
import time
import os
import argparse
import utils

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Video demonstration for Pupil Data and Video')
    parser.add_argument('--data_folder', default='data', type=str, help='pupil data directory for processing gaze information')
    parser.add_argument('--load_json', default=None, type=str, help='option to load json annotation file')
    parser.add_argument('--bucket', default=None, type=int, help='specify bucket name')
    args = parser.parse_args()
    count = 0
    gaze_pos = np.load(os.path.join(args.data_folder,'world_pupil_data.npy'), allow_pickle=True)
    cap = cv2.VideoCapture(os.path.join(args.data_folder,"world.mp4"))
    if args.bucket:
        cap = cv2.VideoCapture(os.path.join(args.data_folder,"world_bucket{}.mp4".format(args.bucket)))
        gaze_pos = np.load(os.path.join(args.data_folder,'world_pupil_data_bucket{}.npy'.format(args.bucket)), allow_pickle=True)
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    print(width, height)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(length)
    frames = None
    if args.load_json:
        tracks = utils.load_json(os.path.join(args.data_folder, args.load_json))
        frames = utils.get_annotations_by_frames(tracks)
        print(frames.keys())
    # Check if camera opened successfully
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")
    
    # cv2.namedWindow('Frame',cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('Frame', width // 3 * 2, height // 3 * 2)
    fourcc = cv2.VideoWriter_fourcc('M','P','E','G')
    writer = cv2.VideoWriter(os.path.join(args.data_folder, "world_annotated{}.avi".format(args.bucket)), fourcc, fps, (width, height))
    # Read until video is completed
    f_index = 0
    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            norm_x, norm_y = gaze_pos[f_index]['norm_pos']
            conf = gaze_pos[f_index]['confidence']
            gaze_x, gaze_y = norm_x * width + 20, (1 - norm_y) * height + 20
            # Display the resulting frame
            if frames:
                if str(f_index) in frames.keys():
                    print("annotating for {}".format(f_index))
                    for annotation in frames[str(f_index)]:
                        outside, id, label, bbox = annotation
                        if outside == "1":
                            continue
                        utils.draw_bbox(frame, bbox, id, '{} {}'.format(label, id))
            cv2.putText(frame, "conf: {}".format(np.round(conf * 100, decimals=2)), (width-250, height-40), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8,(0, 255, 0),1)
            cv2.putText(frame, "topic: {}".format(gaze_pos[f_index]['topic']), (width-250, height-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8,(0, 255, 0),1)
            cv2.circle(frame, (int(gaze_x), int(gaze_y)), 12, (255,165,0), -1)
            cv2.imshow('Frame',frame)
            writer.write(frame)
            f_index += 1
            if f_index >= 5000:
                break
            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
            # Break the loop
        else:
            break
    # When everything done, release the video capture object
    cap.release()
    writer.release()
    # Closes all the frames
    cv2.destroyAllWindows()
