import numpy as np
import cv2
import os
import argparse
import utils

RADIUS = 10
BASE_ALPHA = 0.3
BASE_COLOR = (50/255,205/255,50/255)

def main():
	parser = argparse.ArgumentParser(description='Video demonstration for Pupil Data and Video')
	parser.add_argument('--data_folder', default='data', type=str, help='pupil data directory for processing gaze information')
	parser.add_argument('--bucket', default=None, type=int, help='specify bucket name')
	args = parser.parse_args()

	gaze_pos = np.load(os.path.join(args.data_folder,'world_pupil_data.npy'), allow_pickle=True)
	cap = cv2.VideoCapture(os.path.join(args.data_folder,"world.mp4"))
	if args.bucket:
		cap = cv2.VideoCapture(os.path.join(args.data_folder,"world_bucket{}.mp4".format(args.bucket)))
		gaze_pos = np.load(os.path.join(args.data_folder,'world_pupil_data_bucket{}.npy'.format(args.bucket)), allow_pickle=True)

	width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
	fps = int(cap.get(cv2.CAP_PROP_FPS))
	# initialize the image to just black

	cv2.namedWindow('attention heatmap',cv2.WINDOW_NORMAL)
	cv2.resizeWindow('attention heatmap', width // 3 * 2, height // 3 * 2)
	blank_image = np.zeros((height,width,3), np.uint8)

	for g in gaze_pos:
		x, y = g['norm_pos']
		x, y = int(x * width), int((1-y) * height)
		conf = g['confidence']
		# if conf < 0.5:
		# 	continue
		alpha = BASE_ALPHA * float(conf)
		rgba = (*BASE_COLOR, alpha)
		utils.transparent_circle(blank_image, (x, y), RADIUS, rgba, -1)

	while True:
		cv2.imshow("attention heatmap", blank_image)
		if cv2.waitKey(25) & 0xFF == ord('q'):
			break

	cv2.destroyAllWindows()


if __name__ == '__main__':
	main()