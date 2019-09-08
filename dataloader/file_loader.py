import json
from collections import OrderedDict

def load_json(file_path):
	f = open(file_path, 'r')
	tracks = json.load(f)
	f.close()
	return tracks
'''
input:
	- track data with json formats
output:
	- frames dictionary format:
		... 
'''
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
	print(frames.keys())

FILE_PATH = '../data/outdoor_calibration_outside/OutdoorCalibrationOutside.json'

if __name__ == '__main__':
	test_json_file()