#!/usr/bin/env python3
import sys
import cmath
import logging

import numpy as np
from mpmath import mp
import matplotlib as mpl
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

# see https://www.reddit.com/r/Python/comments/ext3zo/a_very_short_domain_coloring_script
def imshow_color_domain(f):
	xs, xe, rx, ys, ye, ry = -2, 2, 1000, -2, 2, 1000
	x, y = np.ogrid[-xs:xe:1j*rx, ys:ye:1j*ry]
	plt.imshow(np.angle(f((x - 1j*y).T)), cmap=mpl.colormaps['twilight_shifted'])

# adapted from mpmath's default
def colorize(z):
	if cmath.isinf(z):
		return (1.0, 1.0, 1.0)
	if cmath.isnan(z):
		return (0.5, 0.5, 0.5)
	pi = 3.1415926535898
	a = (float(cmath.phase(z)) + pi) / (2*pi)
	a = (a + 0.5) % 1.0
	b = 1.0 - float(1/(1.0+np.abs(z)**0.3))
	#return hls_to_rgb(a, b, 0.8)
	return mpl.colormaps['twilight'](a)[0:3]

def plot_difference_field(f, xmin: float = -5, xmax: float = 5, ymin: float = -5, ymax: float = 5, axes=plt, points=20):
	x, y = np.ogrid[xmin:xmax:1j*points, ymin:ymax:1j*points]
	zs = np.ndarray.flatten(x + 1j * y)
	ds = f(zs) - zs
	xs = [z.real for z in zs]
	ys = [z.imag for z in zs]
	us = np.asarray([z.real for z in ds])
	vs = np.asarray([z.imag for z in ds])
	color = np.sqrt(us**2, vs**2)
	color /= np.max(color)
	color = mpl.colormaps['viridis'](color)
	axes.set_xlabel('Re(z)')
	axes.set_ylabel('Im(z)')
	axes.set_title('Difference Field')
	axes.quiver(xs, ys, us, vs,
				  angles='xy', width=2.5e-3,
				  color=color)

def plot_julia_set(f, xmin=5, xmax=5, ymin=-5, ymax=-5, points=50, axes=plt,
				   iterlimit=10, cutoff=2):
	x, y = np.ogrid[xmin:xmax:1j*points, ymin:ymax:1j*points]
	z0 = (x - y*1j).T
	z = f(z0)
	thresh = cutoff * np.abs(z0)
	image = np.zeros_like(z0, float)
	for step in range(iterlimit, 0, -1):
		image = np.maximum(image, step * (np.abs(z) > thresh))
		z = f(z)
	axes.set_xlabel('Re(z)')
	axes.set_ylabel('Im(z)')
	axes.set_title('Julia Set')
	axes.imshow(image / iterlimit, extent=(xmin, xmax, ymin, ymax), origin='lower')

def plot_quadratic_julia_set(c, xmin=5, xmax=5, ymin=-5, ymax=-5, points=50, axes=plt,
							 iterlimit=10):
	x, y = np.ogrid[xmin:xmax:1j*points, ymin:ymax:1j*points]
	z0 = (x + y*1j).T
	c = complex(c)
	z = z0**2 + c
	thresh = max(abs(c), 2)
	image = np.zeros_like(z0, float)
	for step in range(iterlimit, 0, -1):
		# indicator = np.logical_or(np.isnan(z), np.abs(z) > thresh)
		indicator = np.abs(z) > thresh
		image = np.maximum(image, step * indicator)
		z = z**2 + c
	axes.set_xlabel('Re(z)')
	axes.set_ylabel('Im(z)')
	axes.set_title(r'$Q_c(z) = z^2 + (%.2g %+.2gi)$' % (c.real, c.imag))
	axes.imshow(image / iterlimit, extent=(xmin, xmax, ymin, ymax), origin='lower')

def lerp(a, b, t):
	return b * t + a * (1 - t)

def make_spread(a, b, r, c):
	plt.rcParams.update({'font.size': 10})
	fig, axs = plt.subplots(r, c)

	a = complex(a)
	b = complex(b)
	for row in range(r):
		for col in range(c):
			plot_quadratic_julia_set(lerp(a, b, (c * row + col) / (r * c - 1)),
							xmin=-2, xmax=2, ymin=-2, ymax=2,
							points=500, iterlimit=50, axes=axs[row][col])
			axs[row][col].set_xlabel(None)
			axs[row][col].set_ylabel(None)
	fig.suptitle(r'$c$ on line between $%.2g %+.2gi$ and $%.2g %+.2gi$'
			  % (a.real, a.imag, b.real, b.imag))
	# fig.tight_layout()
	plt.show()

def main():
	make_spread(complex(sys.argv[1]), complex(sys.argv[2]), 3, 7)

if __name__ == '__main__':
	main()
