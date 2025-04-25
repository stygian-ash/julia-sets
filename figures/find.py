#!/usr/bin/env python3
import sys
def lerp(a, b, t):
	return a + (b - a) * t

if __name__ == '__main__':
	pass
	print(
		lerp(-1.5, 0.7, float(eval(sys.argv[1]))/255.),
		lerp(-1.1, 1.1, float(eval(sys.argv[2]))/255.)
	);
