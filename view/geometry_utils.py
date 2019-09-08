import cv2
import shapely
from shapely.geometry import Point, Polygon
from .config import COLORS

def draw_bbox(img, bbox, id, label, col = (0, 255, 0)):

	font = cv2.FONT_HERSHEY_COMPLEX_SMALL
	offset_x, offset_y = 10, 5

	bbox = [float(s) for s in bbox]
	xtl, ytl, xbr, ybr = bbox
	color = COLORS[int(id) % len(COLORS)]
	text_width, text_height = cv2.getTextSize(label, fontFace=font, fontScale=0.8, thickness=1)[0]

	cv2.rectangle(img, (int(xtl), int(ytl - 2*offset_y - text_height)), (int(xtl + 2* offset_x + text_width), int(ytl)), color, -1)
	cv2.putText(img,label, (int(xtl + offset_x), int(ytl - offset_y)), font, 0.8,(0,0,0))
	cv2.rectangle(img, (int(xtl), int(ytl)), (int(xbr), int(ybr)), color, 2)

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