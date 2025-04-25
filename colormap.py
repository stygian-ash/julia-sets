#!/usr/bin/env python3
import sys

from matplotlib._cm_listed import *

def emit_array(colorscheme):
	colors = cmaps[colorscheme].colors
	n = len(colors)
	print(f'const vec4 {colorscheme}[{n}] = vec4[](', end='')
	for i in range(len(colors)):
		color = colors[i]
		end = '' if i == n - 1 else ', '
		print(f'vec4({color[0]}, {color[1]}, {color[2]}, 1.0){end}', end='')
	print(');')

def emit_function(colorscheme):
	n = len(cmaps[colorscheme].colors)
	print('vec4 colorize(float a) {')
	print(f'\treturn {colorscheme}[int(a * {n}.0)];')
	print('}')

if __name__ == '__main__':
	colorscheme = sys.argv[1]
	emit_array(colorscheme)
	# print()
	# emit_function(colorscheme)
