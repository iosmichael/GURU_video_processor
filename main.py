from dataloader import parser, clip, file_loader
from view import geometry_utils
import helper
import cv2
import numpy as np
import glob
import os
import config

def process_praser():
	# preprocess data or check if files are ready
	for DIR in config.DATA_FOLDERS:
		if parser.check_files(os.path.join(config.ROOT_DIR, DIR)):
			parser.parse(os.path.join(config.ROOT_DIR, DIR))
'''
clipping video with number of buckets or seg_length
'''
def process_clipping():
	for DIR in config.DATA_FOLDERS:
		if config.NUM_BUCKET:
			clip.clip(os.path.join(config.ROOT_DIR, DIR), num_bucket=config.NUM_BUCKET)
		if config.SEG_LENGTH:
			clip.clip(os.path.join(config.ROOT_DIR, DIR), num_bucket=config.SEG_LENGTH)
'''
process and save video with overlapping plots
'''
def process_video():
	for DIR in config.DATA_FOLDERS:
		data_folder = os.path.join(config.ROOT_DIR, DIR)
		clip_information = np.load(os.path.join(data_folder, 'clip_information.npy'))
		for entry in clip_information:
			current_dir = os.path.join(data_folder, str(entry['index']))
			j_file = glob.glob(os.path.join(current_dir, '*.json'))
			if len(j_file) == 0:
				print('[VIDEO ERROR] cannot find annotation json file for directory {}'.format(current_dir))
				continue
			if len(j_file) > 1:
				print('[VIDEO WARNING] multiple json files found in the directory (using the first one): {}'.format(os.path.join(data_folder, str(entry['index']))))
			j_file = j_file[0]
			tracks = file_loader.load_json(j_file)
			gaze_pos = np.load(os.path.join(current_dir, 'world_pupil_data_seg.npy'))
			helper.video_process(current_dir, os.path.join(current_dir, 'world_seg.mp4'), tracks, gaze_pos)

'''
print out the attention map from directory
'''
def process_depth():
	for DIR in config.DATA_FOLDERS:
		cap = cv2.VideoCapture(os.path.join(config.ROOT_DIR, DIR, 'world.mp4'))
		width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
		cap.release()
		gaze_pos = np.load(os.path.join(config.ROOT_DIR, DIR, 'world_pupil_data.npy'))
		attention_image = np.zeros((height,width,3), np.uint8)
		for g in gaze_pos:
			x, y = g['norm_pos']
			x, y = int(x * width), int((1-y) * height)
			conf = g['confidence']
			alpha = config.BASE_ALPHA * float(conf)
			rgba = (*config.BASE_COLOR, alpha)
			geometry_utils.transparent_circle(attention_image, (x, y), config.CENTRAL_RADIUS, rgba, -1)
		cv2.imwrite(os.path.join(config.ROOT_DIR, DIR, 'attention_map.png'), attention_image)
		print('[DEPTH INFO] saved attention maps at {}'.format(os.path.join(config.ROOT_DIR, DIR, 'attention_map.png')))

'''
main program
'''
def main():
	if 'parse' in config.MODES:
		# parse the pupil_data
		process_praser()

	if 'clip' in config.MODES:
		# clip the videos into small segments
		process_clipping()

	if 'video' in config.MODES:
		# write video with annotation
		process_video()

	if 'depth' in config.MODES:
		process_depth()

if __name__ == '__main__':
	main()