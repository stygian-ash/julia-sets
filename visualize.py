#!/usr/bin/env python3
import cmath

import numpy as np
from mpmath import mp
import matplotlib as mpl
import matplotlib.pyplot as plt
# from collections.abc import Callable
# 
# type ComplexFn = Callable[[complex], complex]

# see https://www.reddit.com/r/Python/comments/ext3zo/a_very_short_domain_coloring_script
def imshow_color_domain(f):
	xs, xe, rx, ys, ye, ry = -2, 2, 1000, -2, 2, 1000
	x, y = np.ogrid[-xs:xe:1j*rx, ys:ye:1j*ry]
	plt.imshow(np.angle(f((x - 1j*y).T)), cmap=mpl.colormaps['twilight_shifted'])

def quiver_ex():
	# Meshgrid 
	x, y = np.meshgrid(np.linspace(-5, 5, 10),  
					   np.linspace(-5, 5, 10)) 
	  
	# Directional vectors 
	u = -y/np.sqrt(x**2 + y**2) 
	v = x/(x**2 + y**2) 
	  
	# Plotting Vector Field with QUIVER 
	plt.quiver(x, y, u, v, color='g') 
	plt.title('Vector Field') 
	  
	# Setting x, y boundary limits 
	plt.xlim(-7, 7) 
	plt.ylim(-7, 7) 
	  
	# Show plot with grid 
	plt.grid() 
	plt.show() 

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

if __name__ == '__main__':
	fig, axs = plt.subplots(2, 2)
	xmin, xmax = -10, 10
	ymin, ymax = xmin, xmax
	f = lambda z: z**2
	mp.cplot(lambda z: z, points=2e4, axes=axs[0][0],
		  re=[xmin, xmax], im=[ymin, ymax], color=None)
	axs[0][0].set_title('Domain')
	mp.cplot(f, points=2e4, axes=axs[0][1],
		  re=[xmin, xmax], im=[ymin, ymax], color=None)
	axs[0][1].set_title('Range')

	# axs3d = fig.add_subplot(1, 2, 1, projection='3d')
	# mp.splot(lambda x, y: np.abs(f(x + 1j*y)),
	# 	  u=[xmin, xmax], v=[ymin, ymax],
	# 	  axes=axs3d)


	x, y = np.ogrid[xmin:xmax:20j, ymin:ymax:20j]
	zs = np.ndarray.flatten(x + y * 1j)
	ds = f(zs) - zs
	xs = [z.real for z in zs]
	ys = [z.imag for z in zs]
	us = [z.real for z in ds]
	vs = [z.imag for z in ds]
	axs[1][1].set_xlabel('Re(z)')
	axs[1][1].set_ylabel('Im(z)')
	axs[1][1].set_title('Flow Field')
	axs[1][1].quiver(xs, ys, us, vs,
				  angles='xy', width=2.5e-3)

	plt.show()
