from matplotlib import pyplot as plt
from moviepy.video.io.bindings import mplfig_to_npimage
import numpy as np

plt.style.use('dark_background')

# Example: size, _ = get_fig_by_frame(categories, 0, (VIDEO_FRAME_WIDTH-200, 150))
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