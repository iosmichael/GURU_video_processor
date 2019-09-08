import numpy as np
import cv2
import os
import argparse
import utils
from matplotlib import pyplot as plt

CENTRAL_RADIUS = 30
PERIPERAL_RADIUS = 25

def main():
	parser = argparse.ArgumentParser(description='Video demonstration for Pupil Data and Video')
	parser.add_argument('--data_folder', default='data', type=str, help='pupil data directory for processing gaze information')
	parser.add_argument('--load_json', default=None, type=str, help='option to load json annotation file')
	parser.add_argument('--bucket', default=None, type=int, help='specify bucket name')
	args = parser.parse_args()

	gaze_pos = np.load(os.path.join(args.data_folder,'world_pupil_data.npy'), allow_pickle=True)
	cap = cv2.VideoCapture(os.path.join(args.data_folder,"world.mp4"))
	if args.bucket:
		cap = cv2.VideoCapture(os.path.join(args.data_folder,"world_bucket{}.mp4".format(args.bucket)))
		gaze_pos = np.load(os.path.join(args.data_folder,'world_pupil_data_bucket{}.npy'.format(args.bucket)), allow_pickle=True)

	width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
	fps = int(cap.get(cv2.CAP_PROP_FPS))
	length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	# initialize the image to just black
	frames = None
	categories = None
	if args.load_json:
		tracks = utils.load_json(os.path.join(args.data_folder, args.load_json))
		categories = {t['name']:[0] * length for t in tracks['categories']}
		frames = utils.get_annotations_by_frames(tracks)
		print(frames.keys())
		print(categories.keys())

	for f_index, g in enumerate(gaze_pos):
		if str(f_index) in frames.keys():
			x, y = g['norm_pos']
			x, y = int(x * width), int((1-y) * height)
			conf = g['confidence']
			if conf < 0.3:
				continue
			circle = (x, y, 0, CENTRAL_RADIUS)
			for annotation in frames[str(f_index)]:
				outside, id, label, bbox = annotation
				bbox = [float(s) for s in bbox]
				if outside == "1":
					continue
				categories[label][f_index] = utils.intersection_area_circle_rect(circle, bbox)

	remove_keys = []
	for k in categories.keys():
		if sum(categories[k]) == 0:
			remove_keys.append(k)
	
	for k in remove_keys:
		del categories[k]

	fig = plt.figure()
	for k in categories.keys():
		plt.plot(categories[k], label=k)
	plt.legend()
	plt.show()

if __name__ == '__main__':
	main()