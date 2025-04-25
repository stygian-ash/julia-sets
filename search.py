#!/usr/bin/env python3
import visualize
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
	res = 10
	xs, xe, ys, ye = -1, 1, -1, 1
	re, im = np.ogrid[xs:xe:1j*res, ys:ye:1j*res]
	zs = (-re + im*1j).T
	image = np.zeros_like(zs, float)
	for row in range(len(zs)):
		for col in range(len(zs[row])):
			fig, ax = plt.subplots()
			visualize.plot_quadratic_julia_set(zs[row][col],
									  points=500, iterlimit=50,
									  axes=ax)
			plt.show()
			is_connected = input('Connected? ').lower() == 'y'
			image[row][col] = float(is_connected)
			plt.close()

	plt.imshow(image, extent=(xs, xe, ys, ye), origin='lower')
	plt.show()
