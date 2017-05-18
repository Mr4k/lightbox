import numpy as np
import PIL
import matplotlib.pyplot as plt

RESOLUTION = (600, 400)

screenPixels = np.zeros((RESOLUTION[1],RESOLUTION[0], 3))


#camera params
FOV = 85 * np.pi / 180.0
aspect = float(RESOLUTION[0])/float(RESOLUTION[1])
cameraPosition = [0,0,0]


def rayIntersectsSphere(ray, spherePos, sphereRadius, epsilon):
	rad = (2 * np.dot(spherePos, ray)) ** 2 - 4*(np.dot(spherePos,spherePos) - sphereRadius * sphereRadius)
	if rad < 0:
		return -1
	rad = np.sqrt(rad)
	small = (2 * np.dot(spherePos, ray) - rad) / 2.0
	if small >= epsilon:
		return small
	else:
		return (2 * np.dot(spherePos, ray) + rad) / 2.0


def castRay(position, ray, bounce, maxBounce = 4):
	plane = 1
	spherePos = [0,0.4,7]
	sphereRadius = 0.5

	sunDirection = [0.5, 0.7, -1]
	sunDirection = sunDirection / np.linalg.norm(sunDirection)
	skyColor = np.array([138,229,255]) / 255.0
	sunColor = np.array([1,1,1])
	sunStrength = 0.4

	epsilon = 0.01

	surfaceSpecularity = 0.2
	specularitySamples = 10

	intersectionNormal = None

	#for now just check if the ray intersects the plane
	if ray.dot([0,-1,0]) == 0:
		intersectionDepth = -1
	else:
		intersectionDepth = (plane - position[1]) / ray[1]
		intersectionNormal = np.array([0,-1.0,0])
		color = [0.4,0.6,0.6]

	#now check if the plane hits the sphere
	#project point onto ray and check if sphere dist < r
	candidateIntersectionDepth = rayIntersectsSphere(ray, np.array(spherePos) - np.array(position), sphereRadius, epsilon)
	#if candidateIntersectionDepth < 2.6:
	if (candidateIntersectionDepth < intersectionDepth or intersectionDepth < epsilon) and candidateIntersectionDepth > epsilon:
		intersectionDepth = candidateIntersectionDepth
		intersectionNormal = (ray * intersectionDepth - np.array(spherePos))
		intersectionNormal /= np.linalg.norm(intersectionNormal)
		#print intersectionNormal
		color = [0,0,1]


	if intersectionDepth < epsilon:
		return sunColor * sunStrength * (np.dot(ray, sunDirection) / 2.0 + 0.5)**3 + skyColor
	else:
		if bounce == maxBounce:
			return np.array([0,0,0])
		else:
			if bounce > 0:
				specularitySamples = 1
			incomingColor = np.array([0.0,0.0,0.0])
			for i in xrange(specularitySamples):
				nNorm = intersectionNormal
				if specularitySamples > 1:
					mod = np.random.normal(0,0.1 * (1 - surfaceSpecularity)**2,3)
					nNorm += mod
					nNorm /= np.linalg.norm(nNorm)
				newRayDirection = ray + 2 * np.dot(-ray, nNorm) * nNorm
				newRayDirection /= np.linalg.norm(newRayDirection)
				incomingColor += castRay(intersectionDepth * ray, newRayDirection, bounce + 1) / float(specularitySamples)
			shadowColor = castRay(intersectionDepth * ray, -sunDirection, maxBounce)
			#shadowColor = 0
			incomingColor = surfaceSpecularity * incomingColor
			return (incomingColor + shadowColor) * np.array(color)
		#return -intersectionNormal
		#return np.clip(intersectionDepth / 10.0, 0, 1)

#cast the rays
for xx in xrange(RESOLUTION[0]):
	print xx
	for yy in xrange(RESOLUTION[1]):
		ray = np.array([np.tan((xx/float(RESOLUTION[0]) - 0.5) * FOV / 2.0), 
			np.tan((yy/float(RESOLUTION[1]) - 0.5) * FOV / 2.0 * 1.0 / aspect), 1])
		ray /= np.linalg.norm(ray)
		screenPixels[yy, xx] = castRay(cameraPosition, ray, 0)

screenPixels = np.clip(screenPixels,0,1)
screenPixels *= 255

img = PIL.Image.fromarray(np.uint8(screenPixels))

img.save("renderResult.png")

plt.imshow(img)

plt.show()