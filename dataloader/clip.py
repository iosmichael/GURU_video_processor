import cv2
import numpy as np
import time
import argparse
import os

def check_files(dir):
	if not os.path.exists(os.path.join(dir, 'world.mp4')):
		print('[CLIP ERROR] Required: {} not found'.format(os.path.exists(dir, 'world.mp4')))
		return False
	if not os.path.exists(os.path.join(dir, 'world_pupil_data.npy')):
		print('[CLIP ERROR] Required: {} not found'.format(os.path.join(dir, 'world_pupil_data.npy')))
		return False
	print('[CLIP INFO] Find all required files from {}'.format(dir))
	return True

def clip(dir, num_bucket=None, seg_length=None):
	cap = cv2.VideoCapture(os.path.join(dir,'world.mp4'))
	gaze_pos = np.load(os.path.join(dir, 'world_pupil_data.npy'))
	# retrieve video properties
	FRAME_WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	FRAME_HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
	FPS = int(cap.get(cv2.CAP_PROP_FPS))
	LENGTH = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	print('[CLIP INFO] clipping video information: FRAME: ({}, {}), FPS: {}, COUNT: {}'.format(FRAME_WIDTH, FRAME_HEIGHT, FPS, LENGTH))
	assert LENGTH == len(gaze_pos)
	meta_info = {
		"FPS": FPS,
		"FRAME_WIDTH": FRAME_WIDTH,
		"FRAME_HEIGHT": FRAME_HEIGHT
	}
	clip_information = []
	# fire up a video capture
	if num_bucket:
		segment = LENGTH // num_bucket
		for index_bucket in range(num_bucket):
			save_dir = os.path.join(dir, str(index_bucket))
			if not os.path.exists(save_dir):
				os.mkdir(save_dir)

			meta_info['start_frame'] = index_bucket * segment
			seg_length = clip_one_video(save_dir, cap, gaze_pos, segment, meta_info, video_format='mp4')
			clip_information.append({
				"index": index_bucket,
				"owner": os.path.join(dir,'world.mp4'),
				"start_frame": index_bucket * segment,
				"end_frame": index_bucket * segment + seg_length,
				"path": {
					"video": os.path.join(dir, str(index_bucket), "world_seg.mp4"),
					"gaze": os.path.join(dir, str(index_bucket), "world_pupil_data_seg.npy")
				},
				"length": seg_length
			})
			if seg_length < segment and index_bucket != num_bucket - 1:
				print('[CLIP ERROR] clipping videos stop early at {}: {}'.format(index_bucket, seg_length))
				np.save(os.path.join(dir, "clip_information.npy"), clip_information)
				return False
	elif seg_length:
		num_bucket = LENGTH // seg_length
		if LENGTH % seg_length != 0:
			num_bucket += 1
		for index_bucket in range(num_bucket):
			save_dir = os.path.join(dir, str(index_bucket))
			if not os.path.exists(save_dir):
				os.mkdir(save_dir)

			meta_info['start_frame'] = index_bucket * seg_length
			segment_length = clip_one_video(save_dir, cap, gaze_pos, segment, meta_info, video_format='mp4')
			clip_information.append({
				"index": index_bucket,
				"owner": os.path.join(dir,'world.mp4'),
				"start_frame": index_bucket * seg_length,
				"end_frame": index_bucket * seg_length + segment_length,
				"path": {
					"video": os.path.join(dir, str(index_bucket), "world_seg.mp4"),
					"gaze": os.path.join(dir, str(index_bucket), "world_pupil_data_seg.npy")
				},
				"length": seg_length
			})
			if segment_length < seg_length and index_bucket != num_bucket - 1:
				print('[CLIP ERROR] clipping videos stop early at {}: {}'.format(index_bucket, segment_length))
				np.save(os.path.join(dir, "clip_information.npy"), clip_information)
				return False
	else:
		print('[CLIP ERROR] num_bucket and seg_length not setup')

	print('[CLIP INFO] clip dictionary: \n\n{}'.format(clip_information))
	np.save(os.path.join(dir, "clip_information.npy"), clip_information)
	return True

def clip_one_video(dest_dir, cap, gaze_pos, num_iter, meta_info, video_format='mp4'):
	fourcc = cv2.VideoWriter_fourcc(*'XVID')
	FPS, FRAME_WIDTH, FRAME_HEIGHT = meta_info['FPS'], meta_info['FRAME_WIDTH'], meta_info['FRAME_HEIGHT']
	start_frame = meta_info['start_frame']
	writer = cv2.VideoWriter(os.path.join(dest_dir, "world_seg.{}".format(video_format)), fourcc, FPS, (FRAME_WIDTH, FRAME_HEIGHT))
	pupil_data = []
	# here i is frame index
	print('[CLIP INFO] clipping one segment from frame: {}'.format(start_frame))
	for i in range(start_frame, start_frame + num_iter):
		ret, frame = cap.read()
		if not ret:
			np.save(os.path.join(dest_dir, "world_pupil_data_seg.npy"), pupil_data)
			writer.release()
			return i
		# height, width, _ = frame.shape
		norm_x, norm_y = gaze_pos[i]['norm_pos']
		conf = gaze_pos[i]['confidence']
		# Display the resulting frame
		pupil_data.append({
			'norm_pos':gaze_pos[i]['norm_pos'],
			'confidence': gaze_pos[i]['confidence'],
			'topic':gaze_pos[i]['topic']
			})
		writer.write(frame)
	np.save(os.path.join(dest_dir, "world_pupil_data_seg.npy"), pupil_data)	
	writer.release()
	return num_iter


'''
old usage: clipping data with number of buckets
'''
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
					'confidence': gaze_pos[f_index]['confidence'],
					'topic':gaze_pos[f_index]['topic']
					})
				writer.write(frame)
				f_index += 1
				# Press Q on keyboard to  exit
				if cv2.waitKey(25) & 0xFF == ord('q'):
					break
			else:
				print('End of frame, exiting the program')
				break
		# When everything done, release the video capture object
		cap.release()
		np.save(os.path.join(args.data_folder, "world_pupil_data_bucket{}.npy".format(bucket)), data)
		writer.release()
		# Closes all the frames
		cv2.destroyAllWindows()
