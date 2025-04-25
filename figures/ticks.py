#!/usr/bin/env python3
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys

if __name__ == '__main__':
	img = mpimg.imread(sys.argv[1])

	plt.rcParams.update({"ytick.color" : "w",
						 "xtick.color" : "w",
						 "axes.labelcolor" : "w",
						 "axes.edgecolor" : "w"})

	imgplot = plt.imshow(img,
					  extent=(-1.5, .7, -1.1, 1.1))

	if len(sys.argv) >= 3:
		plt.title(sys.argv[2], size=24, color='white')
	plt.title(r'$\{c \in \mathbf{C} : 0 \in K(Q_c)\}$', size=24, color='white')
	plt.xlabel(r'$\operatorname{Re}(c)$', color='white')
	plt.ylabel(r'$\operatorname{Im}(c)$', color='white')

	if len(sys.argv) >= 4:
		plt.savefig(sys.argv[3], transparent=True)
	else:
		plt.show()
