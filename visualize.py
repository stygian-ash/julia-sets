#!/usr/bin/env python3
import re
import sys
import math
import cmath
import logging
import itertools
import multiprocess
from multiprocess import Pool

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
	def plot(ax):
		ax.set_xlabel('Re(z)')
		ax.set_ylabel('Im(z)')
		ax.set_title('Julia Set')
		ax.imshow(image / (1 if binary else iterlimit), extent=(xmin, xmax, ymin, ymax), origin='lower')
	if axes is not None:
		plot(axes)
	return plot

def texify(x, n=2, plus=False):
	s = f'%{"+" if plus else ""}.{n}g' % x
	if 'e' not in s:
		return s
	_, frac, exp = re.match(r'(\d+)e(-?\d+)', s)
	if float(frac) == 1:
		return f'10^{exp}'
	return fr'{frac} \times 10^{exp}'

def format_complex(z, n=2, plus=True, ε=1e-12):
	return '%+0.3g %+0.3gi' % (z.real, z.imag)
	if abs(z.imag) < ε:
		if abs(z.real) < ε:
			return '+0' if plus else '0'
		return texify(z.real, n, plus)
	if abs(z.real < ε):
		if abs(z.imag - 1) < ε:
			return '+i' if plus else 'i'
		if abs(z.imag + 1) < ε:
			return '-i' if plus else 'i'
		return texify(z.imag, n, plus) + 'i'
	if abs(z.imag - 1) < ε:
		return texify(z.real, n, plus) + '+i'
	return texify(z.real, n, plus) + texify(z.imag, n, True) + 'i'


def plot_quadratic_julia_set(c, xmin=-2, xmax=2, ymin=-2, ymax=2, points=1000, axes=plt,
							 iterlimit=50, binary=False):
	x, y = np.ogrid[xmin:xmax:1j*points,ymin:ymax:1j*points]
	z0 = (x + y*1j).T
	c = complex(c)
	z = z0**2 + c
	# indicator = np.logical_or(np.isnan(z), np.abs(z) > thresh)
	# image = np.maximum(image, (1 if binary else step) * indicator)
	thresh = max(abs(c), 2)
	image = np.zeros_like(z0, float)
	for step in range(iterlimit, 0, -1):
		indicator = np.abs(z) > thresh
		image = np.maximum(image, step * indicator)
		z = z**2 + c
	def plot(axes):
		axes.xlabel('Re(z)')
		axes.ylabel('Im(z)')
		axes.title(r'$Q_c(z) = z^2 %s$' % (format_complex(c, 3)))
		axes.imshow(image / iterlimit, extent=(xmin, xmax, ymin, ymax), origin='lower')
		# axes.imshow(image / (1 if binary else iterlimit), extent=(xmin, xmax, ymin, ymax), origin='lower')
	if axes is not None:
		plot(axes)
	return plot

def lerp(a, b, t):
	return b * t + a * (1 - t)

def naive_connected_heuristic(c, r=0.25, n=8, N=50):
	for i in range(n):
		θ = lerp(0, 2*math.pi, i / n)
		z0 = r*(np.cos(θ) + 1j * np.sin(θ))
		z = z0
		thresh = max(abs(c), 2)
		for k in range(N):
			z = z**2 + c
			if z > thresh:
				return k / N
	return 1

def make_spread(a, b, r, c, ease=lambda z: z, overlap=True, shade=lambda r, c: 0,
				xmin=-2, xmax=2, ymin=-2, ymax=2, threads=1,
				points=1000, iterlimit=100):
	plt.rcParams.update({'font.size': 10})
	fig, axs = plt.subplots(r, c)

	a = complex(a)
	b = complex(b)

	def get_plot_fn(row, col):
		ret = plot_quadratic_julia_set(ease(lerp(a, b, (c * row + col) / (r * c - int(overlap))))
								  + shade(row / (r - int(overlap)), 1 - col / (c - int(overlap))),
						   xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
						   points=points, iterlimit=iterlimit, axes=plt)

		print('.', end='', flush=True)
		return ret

	if threads > 1:
		with Pool(threads) as p:
			subplots = p.starmap(get_plot_fn, itertools.product(range(r), range(c)))
	else:
		subplots = itertools.starmap(get_plot_fn, itertools.product(range(r), range(c)))

	it = iter(subplots)
	for row in range(r):
		for col in range(c):
			next(it)(axs[row][col])
			axs[row][col].set_xlabel(None)
			axs[row][col].set_ylabel(None)
	fig.suptitle(r'$c$ on line between $%s$ and $%s$'
			  % (format_complex(a, 2), format_complex(b, 2)))
	# fig.tight_layout()
	return fig, axs

def main():
	# fig, ax = plt.subplots()
	# plot_quadratic_julia_set(1j,
	# 					  xmin=-2, xmax=2, ymin=-2, ymax=2,
	# 					  points=5000, iterlimit=50, axes=ax)
	# make_spread(complex(sys.argv[1]), complex(sys.argv[2]), 3, 7)
	# ε = 0.01
	# θ = 11 * math.pi / 12
	# fig, axs = make_spread(θ - ε, θ + ε, 4, 6, lambda θ: np.cos(θ) + np.sin(θ) * 1j, False,
	# 					xmin=-0.5, xmax=0.5, ymin=-0.5, ymax=0.5)
	# for row in axs:
	# 	for subfig in row:
	# 		subfig.set_title(None)
	# fig.suptitle('near transition point')
	fig, axs = make_spread(0, 0, 8, 12,
						shade=lambda r, c: lerp(-1, 1, c) + lerp(0.5, 1, r) * 1j,
						xmin=-2, xmax=2, ymin=-2, ymax=2,
						points=100, iterlimit=50)
	for row in axs:
		for f in row:
			f.set_title(None)
	fig.suptitle('$c$ on lattice')
	# fig, ax = plt.subplots()
	# plot_quadratic_julia_set(-1,
	# 					  # xmin=-2.5, xmax=2.5, ymin=0, ymax=10,
	# 					  points=500, iterlimit=50, axes=ax)
	# plt.show()
	# fig, axs = plt.subplots()
	# res = 2000
	# xs, xe, ys, ye = -0.75, 1.6, -1.2, 1.2
	# x, y = np.ogrid[xs:xe:1j*res, ys:ye:1j*res]
	# zs = (-x + y*1j).T
	# image = np.zeros_like(zs, float)
	# 
	# for row in range(len(zs)):
	# 	for col in range(len(zs[row])):
	# 		image[row][col] = 1 - naive_connected_heuristic(zs[row][col], r=0.5, N=100)
	# axs.set_xlabel('Re($z$)')
	# axs.set_ylabel('Im($z$)')
	# axs.set_title('Plot of $c$ for which $J_c(Q_c)$ is 2-connected')
	# axs.imshow(image, extent=(xs, xe, ys, ye), origin='lower')
	plt.show()

if __name__ == '__main__':
	main()
