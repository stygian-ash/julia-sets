#!/usr/bin/env python3
import re
import sys
import math
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
				   iterlimit=10, cutoff=2, binary=False):
	x, y = np.ogrid[xmin:xmax:1j*points, ymin:ymax:1j*points]
	z0 = (x - y*1j).T
	z = f(z0)
	thresh = cutoff * np.abs(z0)
	image = np.zeros_like(z0, float)
	for step in range(iterlimit, 0, -1):
		image = np.maximum(image, (1 if binary else step) * (np.abs(z) > thresh))
		z = f(z)
	axes.set_xlabel('Re(z)')
	axes.set_ylabel('Im(z)')
	axes.set_title('Julia Set')
	axes.imshow(image / (1 if binary else iterlimit), extent=(xmin, xmax, ymin, ymax), origin='lower')

def texify(x, n=2, plus=False):
	s = f'%{"+" if plus else ""}.{n}g' % x
	if 'e' not in s:
		return s
	_, frac, exp = re.match(r'(\d+)e(-?\d+)', s)
	if float(frac) == 1:
		return f'10^{exp}'
	return fr'{frac} \times 10^{exp}'

def format_complex(z, n=2, paren=False, ε=1e-12):
	if abs(z.imag) < ε:
		if abs(z.real) < ε:
			return '+0'
		return texify(z.real, n, True)
	if abs(z.real < ε):
		return texify(z.imag, n, True) + 'i'
	return texify(z.real, n, True) + texify(z.imag, n, True) + 'i'


def plot_quadratic_julia_set(c, xmin=5, xmax=5, ymin=-5, ymax=-5, points=50, axes=plt,
							 iterlimit=10, binary=False):
	x, y = np.ogrid[xmin:xmax:1j*points, ymin:ymax:1j*points]
	z0 = (x + y*1j).T
	c = complex(c)
	z = z0**2 + c
	thresh = max(abs(c), 2)
	image = np.zeros_like(z0, float)
	for step in range(iterlimit, 0, -1):
		# indicator = np.logical_or(np.isnan(z), np.abs(z) > thresh)
		indicator = np.abs(z) > thresh
		image = np.maximum(image, (1 if binary else step) * indicator)
		z = z**2 + c
	axes.set_xlabel('Re(z)')
	axes.set_ylabel('Im(z)')
	axes.set_title(r'$Q_c(z) = z^2 %s$' % (format_complex(c, 3)))
	axes.imshow(image / (1 if binary else iterlimit), extent=(xmin, xmax, ymin, ymax), origin='lower')

def lerp(a, b, t):
	return b * t + a * (1 - t)

def make_spread(a, b, r, c, ease=lambda z: z, overlap=True):
	plt.rcParams.update({'font.size': 10})
	fig, axs = plt.subplots(r, c)

	a = complex(a)
	b = complex(b)
	for row in range(r):
		for col in range(c):
			plot_quadratic_julia_set(ease(lerp(a, b, (c * row + col) / (r * c - int(overlap)))),
							xmin=-2, xmax=2, ymin=-2, ymax=2,
							points=1000, iterlimit=100, axes=axs[row][col])
			axs[row][col].set_xlabel(None)
			axs[row][col].set_ylabel(None)
	fig.suptitle(r'$c$ on line between $%s$ and $%s$'
			  % (format_complex(a, 2), format_complex(b, 2)))
	# fig.tight_layout()
	return fig, axs

def main():
	fig, ax = plt.subplots()
	plot_quadratic_julia_set(1j,
						  xmin=-2, xmax=2, ymin=-2, ymax=2,
						  points=5000, iterlimit=50, axes=ax)
	# make_spread(complex(sys.argv[1]), complex(sys.argv[2]), 3, 7)
	# fig, axs = make_spread(0, 2*math.pi, 3, 6, lambda θ: np.cos(θ) + np.sin(θ) * 1j, False)
	# fig.suptitle('$c$ on unit circle')
	plt.show()

if __name__ == '__main__':
	main()
