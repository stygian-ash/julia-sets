#!/usr/bin/env python3
import sys
from math import *

import numpy as np
import matplotlib.pyplot as plt

import visualize

if __name__ == '__main__':
	if len(sys.argv) < 2:
		print('Usage: ./render.py <c-expression>')
		exit(1)
	c = eval(sys.argv[1])
	fig, ax = plt.subplots()
	visualize.plot_quadratic_julia_set(
		c, points=5000, iterlimit=50, axes=ax)
	plt.show()
