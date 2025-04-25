#!/usr/bin/env python3
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys

if __name__ == '__main__':
	img = mpimg.imread(sys.argv[1])
	imgplot = plt.imshow(img,
					  extent=(-1.5, .7, -1.1, 1.1))
	if len(sys.argv) >= 3:
		plt.title(sys.argv[2], fontsize=24)
	plt.xlabel(r'$\operatorname{Re}(c)$')
	plt.ylabel(r'$\operatorname{Im}(c)$')
	if len(sys.argv) >= 4:
		plt.savefig(sys.argv[3])
	else:
		plt.show()
