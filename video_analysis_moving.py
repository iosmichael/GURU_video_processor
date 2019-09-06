import cv2, os
from moviepy.video.io.bindings import mplfig_to_npimage
from matplotlib import pyplot as plt
import utils
import numpy as np


FILE_DIR = './outdoor_calibration_outside'
JSON_FILE = 'calibration2.json'
MOVIE_FILE = 'world_bucket2.mp4'
GAZE_FILE = 'world_pupil_data_bucket2.npy'

graph_alpha = 0.6

CENTRAL_RADIUS = 20
BASE_COLOR = (50/255,205/255,50/255)

def load_graph_data(gaze_pos, tracks, frame_size):
	width, height = frame_size
	frames = utils.get_annotations_by_frames(tracks)
	categories = {t['name']:[0] * len(gaze_pos) for t in tracks['categories']}
	# LOAD GRAPH DATA
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
				categories[label][f_index] = max(utils.intersection_area_circle_rect(circle, bbox), categories[label][f_index])

	remove_keys = []
	for k in categories.keys():
		if sum(categories[k]) == 0:
			remove_keys.append(k)
	# Remove unnecessary key from the category
	for k in remove_keys:
		del categories[k]

	return categories, frames

def get_fig_by_frame(categories, f_index, f_size):
	f_dpi = 100
	f_w, f_h = f_size
	fig = plt.figure(figsize=(f_w / f_dpi, f_h / f_dpi), dpi=f_dpi)
	ax = fig.add_subplot(111)
	for k in categories.keys():
		ax.plot(categories[k], label=k)
		ax.fill_between(np.arange(len(categories[k])), 0, categories[k], alpha=0.4)
	ax.legend()
	ax.set_ylabel('Area %')
	ax.spines['top'].set_visible(False)
	ax.spines['right'].set_visible(False)
	ax.spines['left'].set_visible(False)
	ax.grid(axis='y')
	[t.set_visible(False) for t in ax.get_xticklines()]
	# draw vertical line
	ax.axvline(x=f_index)
	fig.tight_layout()
	imfig = mplfig_to_npimage(fig)
	plt.close(fig)
	h, w, _ = imfig.shape
	return (h, w), imfig

def main():
	tracks = utils.load_json(os.path.join(FILE_DIR, JSON_FILE))
	cap = cv2.VideoCapture(os.path.join(FILE_DIR, MOVIE_FILE))
	width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
	fps, length = int(cap.get(cv2.CAP_PROP_FPS)), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	print("Movie Resolution: {} x {}, FPS: {}, Length: {}".format(width, height, fps, length))
	gaze_pos = np.load(os.path.join(FILE_DIR, GAZE_FILE), allow_pickle=True)
	
	categories, frames = load_graph_data(gaze_pos, tracks, (width, height))
	plt.style.use('dark_background')
	size, _ = get_fig_by_frame(categories, 0, (width-200, 150))
	gh, gw = size
	f_index = 0
	cv2.namedWindow('Frame',cv2.WINDOW_NORMAL)
	cv2.resizeWindow('Frame', width // 3 * 2, height // 3 * 2)

	fourcc = cv2.VideoWriter_fourcc(*'XVID')
	writer = cv2.VideoWriter(os.path.join(FILE_DIR, "world_annotated3_graph.mp4"), fourcc, fps, (width, height))

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
						utils.draw_bbox(frame, bbox, id, '{} {}'.format(label, id))

			utils.transparent_circle(frame, (gaze_x, gaze_y), CENTRAL_RADIUS, (*BASE_COLOR, 0.8), -1)

			_, imfig = get_fig_by_frame(categories, f_index, (width-200, 150))
			frame[height-gh:, :gw,:] = imfig * graph_alpha + frame[height-gh:, :gw,:] * (1-graph_alpha)
			frame[height-gh:, gw:, :] = np.zeros((gh, width-gw, 3), np.uint8) * graph_alpha + frame[height-gh:, gw:, :] * (1-graph_alpha)
			cv2.putText(frame, "conf: {}".format(np.round(conf * 100, decimals=2)), (width-180, height-80), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8,(255, 255, 255),1)
			cv2.putText(frame, "topic: {}".format(gaze_pos[f_index]['topic']), (width-180, height-40), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8,(255, 255, 255),1)
			cv2.imshow('Frame',frame)
			writer.write(frame)
			f_index += 1
			if f_index >= length:
			    break
			# Press Q on keyboard to  exit
			if cv2.waitKey(25) & 0xFF == ord('q'):
				break
		# Break the loop
		else:
			break

	cap.release()
	writer.release()
	cv2.destroyAllWindows()

if __name__ == '__main__':
	main()