import numpy as np
import PIL
import matplotlib.pyplot as plt

RESOLUTION = (700, 500)

screenPixels = np.zeros((RESOLUTION[1],RESOLUTION[0], 3))


#camera params
FOV = 85 * np.pi / 180.0
aspect = float(RESOLUTION[0])/float(RESOLUTION[1])
cameraPosition = [0,0,0]


def rayIntersectsSphere(ray, spherePos, sphereRadius):
	rad = (2 * np.dot(spherePos, ray)) ** 2 - 4*(np.dot(spherePos,spherePos) - sphereRadius)
	if rad < 0:
		return -1
	rad = np.sqrt(rad)
	small = (2 * np.dot(spherePos, ray) - rad) / 2.0
	if small >= 0:
		return small
	else:
		return (2 * np.dot(spherePos, ray) + rad) / 2.0


def castRay(position, ray):
	plane = 1
	spherePos = [0,0,5]
	sphereRadius = 0.5

	#for now just check if the ray intersects the plane
	if ray.dot([0,1,0]) == 0:
		intersectionDepth = -1
	else:
		intersectionDepth = plane / ray[1]

	#now check if the plane hits the sphere
	#project point onto ray and check if sphere dist < r
	candidateIntersectionDepth = rayIntersectsSphere(ray, spherePos, sphereRadius)
	#if candidateIntersectionDepth < 2.6:
	if (candidateIntersectionDepth < intersectionDepth or intersectionDepth < 0) and candidateIntersectionDepth > 0:
		intersectionDepth = candidateIntersectionDepth
		return [0,0,np.clip(1 - np.linalg.norm(intersectionDepth * ray) / 8.0, 0, 1)]


	if intersectionDepth < 0:
		return[0,0,0]
	else:
		return np.clip(1 - np.linalg.norm(intersectionDepth * ray) / 8.0, 0, 1)
		#return np.clip(intersectionDepth / 10.0, 0, 1)

#cast the rays
for xx in xrange(RESOLUTION[0]):
	for yy in xrange(RESOLUTION[1]):
		ray = np.array([np.tan((xx/float(RESOLUTION[0]) - 0.5) * FOV / 2.0), 
			np.tan((yy/float(RESOLUTION[1]) - 0.5) * FOV / 2.0 * 1.0 / aspect), 1])
		ray /= np.linalg.norm(ray)
		screenPixels[yy, xx] = castRay(cameraPosition, ray)


screenPixels *= 255

img = PIL.Image.fromarray(np.uint8(screenPixels))

plt.imshow(img)

plt.show()