import numpy as np
import PIL
import matplotlib.pyplot as plt

RESOLUTION = (700, 500)

screenPixels = np.zeros((RESOLUTION[1],RESOLUTION[0], 3))


#camera params
FOV = 60 * np.pi / 180.0
aspect = float(RESOLUTION[0])/float(RESOLUTION[1])
cameraPosition = [0,0,0]


def castRay(position, ray):
	plane = 1.0

	#for now just check if the ray intersects the plane
	if ray.dot([0,1,0]) == 0:
		return [0,0,0]
	intersectionDepth = plane / ray[1]

	if intersectionDepth < 0:
		return[0,0,0]
	else:
		return [0.5,0.5,0.5]

#cast the rays
for xx in xrange(RESOLUTION[0]):
	for yy in xrange(RESOLUTION[1]):
		screenPixels[yy, xx] = castRay(cameraPosition, np.array([np.tan((xx/float(RESOLUTION[0]) - 0.5) * FOV / 2.0), 
			np.tan((yy/float(RESOLUTION[1]) - 0.5) * FOV / 2.0 * 1.0 / aspect), 1]))


screenPixels *= 255

img = PIL.Image.fromarray(np.uint8(screenPixels))

plt.imshow(img)

plt.show()