
# three modes: parse, clip, video, and depth
# MODES = ['parse', 'clip', 'video']
MODES = ['video']

# Data Options
ROOT_DIR = './data'
DATA_FOLDERS = ['outdoor_calibration_outside']
# Parse Mode


# Clip Mode
NUM_BUCKET = 2
SEG_LENGTH = None

# Video Mode
GRAPH_ALPHA = 0.6

CENTRAL_RADIUS = 20
BASE_COLOR = (50/255,205/255,50/255)
BASE_ALPHA = 0.4

# Player Options
WRITE_VIDEO = True
WITH_PREVIEW = True


# Depth Options
DEPTH_FOLDERS = ['outdoor_calibration_outside']