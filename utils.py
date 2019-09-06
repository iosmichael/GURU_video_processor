import json
import cv2
import math
import shapely
from collections import OrderedDict
from shapely.geometry import Point, Polygon

FILE_PATH = './outdoor_calibration_outside/OutdoorCalibrationOutside.json'

COLORS = [(0,102,255), (175,89,62), (1,163,104), (255,134,31), (237,10,63), (255,63,52), (118,215,234),
		(131,89,163), (251,232,112), (197,225,122), (3,187,133), (255,223,0),] 
		# '#8B8680', '#0A6B0D',
		# '#8FD8D8', '#A36F40', '#F653A6', '#CA3435', '#FFCBA4', '#FF99CC', '#FA9D5A',
		# '#FFAE42', '#A78B00', '#788193', '#514E49', '#1164B4', '#F4FA9F', '#FED8B1',
		# '#C32148', '#01796F', '#E90067', '#FF91A4', '#404E5A', '#6CDAE7', '#FFC1CC',
		# '#006A93', '#867200', '#E2B631', '#6EEB6E', '#FFC800', '#CC99BA', '#FF007C',
		# '#BC6CAC', '#DCCCD7', '#EBE1C2', '#A6AAAE', '#B99685', '#0086A7', '#5E4330',
		# '#C8A2C8', '#708EB3', '#BC8777', '#B2592D', '#497E48', '#6A2963', '#E6335F',
		# '#00755E', '#B5A895', '#0048ba', '#EED9C4', '#C88A65', '#FF6E4A', '#87421F',
		# '#B2BEB5', '#926F5B', '#00B9FB', '#6456B7', '#DB5079', '#C62D42', '#FA9C44',
		# '#DA8A67', '#FD7C6E', '#93CCEA', '#FCF686', '#503E32', '#FF5470', '#9DE093',
		# '#FF7A00', '#4F69C6', '#A50B5E', '#F0E68C', '#FDFF00', '#F091A9', '#FFFF66',
		# '#6F9940', '#FC74FD', '#652DC1', '#D6AEDD', '#EE34D2', '#BB3385', '#6B3FA0',
		# '#33CC99', '#FFDB00', '#87FF2A', '#6EEB6E', '#FFC800', '#CC99BA', '#7A89B8',
		# '#006A93', '#867200', '#E2B631', '#D9D6CF']

def load_json(file_path):
	f = open(file_path, 'r')
	tracks = json.load(f)
	f.close()
	return tracks

def draw_bbox(img, bbox, id, label, col = (0, 255, 0)):
	xtl, ytl, xbr, ybr = bbox
	color = COLORS[int(id) % len(COLORS)]
	xtl, ytl, xbr, ybr = float(xtl), float(ytl), float(xbr), float(ybr)
	cv2.rectangle(img, (int(xtl), int(ytl)), (int(xbr), int(ybr)), color, 2)
	cv2.putText(img,label.upper(), (int(xtl),int(ytl)-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,(0,255,0))

# input color: rgba(...)
def transparent_circle(img, center, radius, color, thickness):
	center = tuple(map(int, center))
	assert len(color) == 4 and all(type(c) == float and 0.0 <= c <= 1.0 for c in color)
	bgr = [255 * c for c in color[:3]]  # convert to 0-255 scale for OpenCV
	alpha = color[-1]
	radius = int(radius)
	if thickness > 0:
		pad = radius + 2 + thickness
	else:
		pad = radius + 3
	roi = (
		slice(center[1] - pad, center[1] + pad),
		slice(center[0] - pad, center[0] + pad),
	)

	try:
		overlay = img[roi].copy()
		cv2.circle(img, center, radius, bgr, thickness=thickness, lineType=cv2.LINE_AA)
		opacity = alpha
		cv2.addWeighted(
			src1=img[roi],
			alpha=opacity,
			src2=overlay,
			beta=1.0 - opacity,
			gamma=0,
			dst=img[roi],
		)
	except:
		logger.debug(
			"transparent_circle would have been partially outside of img. Did not draw it."
		)

# bbox = (xtl, ytl, xbr, ybr)
# circle = (center_x, center_y, radius_in, radius_out)
def intersection_area_circle_rect(circle, bbox):
	xtl, ytl, xbr, ybr = bbox
	center_x, center_y, radius_in, radius_out = circle
	c1, c2 = Point((center_x, center_y)).buffer(radius_in), Point((center_x, center_y)).buffer(radius_out)
	box = Polygon([(xtl, ytl), (xbr, ytl), (xbr, ybr), (xtl, ybr)])
	c = c2.difference(c1)
	return (c.intersection(box)).area / c.area

def get_annotations_by_frames(tracks):
	frames = OrderedDict()
	start_frame, stop_frame = int(tracks['meta']['task']['start_frame']), int(tracks['meta']['task']['stop_frame'])
	for item in tracks['tracks']:
		for annotation in item['shapes']:
			if annotation['frame'] not in frames.keys():
				frames[annotation['frame']] = []
			frames[annotation['frame']].append((annotation['outside'], item['id'], item['label'], (annotation['xtl'], annotation['ytl'], annotation['xbr'], annotation['ybr'])))
	return frames

def test_json_file():
	tracks = load_json(FILE_PATH)
	frames = get_annotations_by_frames(tracks)
	print(frames)

def test_intersection_area():
	bbox = (0, 0, 10, 10)
	circle = (5, 5, 0, 6)
	print(intersection_area_circle_rect(circle, bbox))

def main():
	test_intersection_area()

if __name__ == '__main__':
	main()
