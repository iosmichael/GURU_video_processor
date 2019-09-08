import config
from view import geometry_utils, plot_utils
from dataloader import file_loader
import numpy as np
import os
import cv2

def load_graph_data(gaze_pos, tracks, frame_size):
	width, height = frame_size
	frames = file_loader.get_annotations_by_frames(tracks)
	categories = {t['name']:[0] * len(gaze_pos) for t in tracks['categories']}
	# LOAD GRAPH DATA
	for f_index, g in enumerate(gaze_pos):
		if str(f_index) in frames.keys():
			x, y = g['norm_pos']
			x, y = int(x * width), int((1-y) * height)
			conf = g['confidence']
			if conf < 0.3:
				continue
			circle = (x, y, 0, config.CENTRAL_RADIUS)
			for annotation in frames[str(f_index)]:
				outside, id, label, bbox = annotation
				bbox = [float(s) for s in bbox]
				if outside == "1":
					continue
				categories[label][f_index] = max(geometry_utils.intersection_area_circle_rect(circle, bbox), categories[label][f_index])

	remove_keys = []
	for k in categories.keys():
		if sum(categories[k]) == 0:
			remove_keys.append(k)
	# Remove unnecessary key from the category
	for k in remove_keys:
		del categories[k]

	return categories, frames

def video_process(dir, video_path, tracks, gaze_pos):
	# tracks = utils.load_json(os.path.join(FILE_DIR, JSON_FILE))
	cap = cv2.VideoCapture(video_path)
	width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
	fps, length = int(cap.get(cv2.CAP_PROP_FPS)), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	print("[VIDEO INFO] Movie Resolution: {} x {}, FPS: {}, Length: {}".format(width, height, fps, length))
	categories, frames = load_graph_data(gaze_pos, tracks, (width, height))
	# plt.style.use('dark_background')
	size, _ = plot_utils.get_fig_by_frame(categories, 0, (width-200, 150))
	gh, gw = size
	f_index = 0
	# cv2.namedWindow('Frame',cv2.WINDOW_NORMAL)
	# cv2.resizeWindow('Frame', width // 3 * 2, height // 3 * 2)

	fourcc = cv2.VideoWriter_fourcc(*'XVID')
	writer = cv2.VideoWriter(os.path.join(dir, "annotated_video.mp4"), fourcc, fps, (width, height))

	while(cap.isOpened()):
		# Capture frame-by-frame
		ret, frame = cap.read()
		if ret == True:
			norm_x, norm_y = gaze_pos[f_index]['norm_pos']
			conf = gaze_pos[f_index]['confidence']
			gaze_x, gaze_y = norm_x * width, (1 - norm_y) * height
			# Display the resulting frame
			if frames:
				if str(f_index) in frames.keys():
					for annotation in frames[str(f_index)]:
						outside, id, label, bbox = annotation
						if outside == "1":
							continue
						geometry_utils.draw_bbox(frame, bbox, id, '{} {}'.format(label, id))

			geometry_utils.transparent_circle(frame, (gaze_x, gaze_y), config.CENTRAL_RADIUS, (*config.BASE_COLOR, 0.8), -1)

			_, imfig = plot_utils.get_fig_by_frame(categories, f_index, (width-200, 150))
			frame[height-gh:, :gw,:] = imfig * config.GRAPH_ALPHA + frame[height-gh:, :gw,:] * (1-config.GRAPH_ALPHA)
			frame[height-gh:, gw:, :] = np.zeros((gh, width-gw, 3), np.uint8) * config.GRAPH_ALPHA + frame[height-gh:, gw:, :] * (1-config.GRAPH_ALPHA)
			cv2.putText(frame, "conf: {}".format(np.round(conf * 100, decimals=2)), (width-180, height-80), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8,(255, 255, 255),1)
			cv2.putText(frame, "model: {}".format(gaze_pos[f_index]['topic']), (width-180, height-40), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8,(255, 255, 255),1)
			# cv2.imshow('Frame',frame)
			writer.write(frame)
			f_index += 1
			# Press Q on keyboard to  exit
			if cv2.waitKey(25) & 0xFF == ord('q'):
				break
		# Break the loop
		else:
			break
	cap.release()
	writer.release()
	cv2.destroyAllWindows()