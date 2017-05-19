import numpy as np
import PIL
import matplotlib.pyplot as plt
import noise

RESOLUTION = (1280, 720)

screenPixels = np.zeros((RESOLUTION[1],RESOLUTION[0], 3))

#camera params
FOV = 85 * np.pi / 180.0
aspect = float(RESOLUTION[0])/float(RESOLUTION[1])
cameraPosition = [0,0,0]

############
# Generic Classes to build renderables
############

class Material:
	surfaceSpecularity = 0.15
	color = [1,1,1]

	def defaultSampleTexture(self, pos):
		return self.color

	sampleTexture = defaultSampleTexture

class Geom:
	def hitsRay(self, ray, cameraPos):
		return (-1, None)

class Light:
	strength = 0.4
	color = 0.4

###########
# Specific Renderables and Materials (maybe move to a different file later)
###########

class Plane(Geom):
	def __init__(self, y):
		self.y = y

	def hitsRay(self, ray, cameraPos, epsilon):
		intersectionNormal = None
		if ray.dot([0,-1,0]) == 0:
			intersectionDepth = -1
		else:
			intersectionDepth = (self.y - cameraPos[1]) / ray[1]
			intersectionNormal = np.array([0,-1.0,0])
		return (intersectionDepth, intersectionNormal)

class Sphere(Geom):
	def __init__(self, spherePosition, sphereRadius):
		self.spherePosition = spherePosition
		self.sphereRadius = sphereRadius

	def hitsRay(self, ray, cameraPos, epsilon):
		spherePos = self.spherePosition - cameraPos
		rad = (2 * np.dot(spherePos, ray)) ** 2 - 4*(np.dot(spherePos,spherePos) - self.sphereRadius * self.sphereRadius)
		if rad < 0:
			return (-1, None)
		rad = np.sqrt(rad)
		small = (2 * np.dot(spherePos, ray) - rad) / 2.0
		if small >= epsilon:
			intersectionNormal = (ray * small - spherePos)
			intersectionNormal /= np.linalg.norm(intersectionNormal)
			return (small, intersectionNormal)
		else:
			big = (2 * np.dot(spherePos, ray) + rad) / 2.0
			intersectionNormal = (ray * big - spherePos)
			intersectionNormal /= np.linalg.norm(intersectionNormal)
			return (big, intersectionNormal)

def castRay(world, position, ray, bounce, maxBounce = 4):
	plane = 1
	spherePos = [0,0.4,7]
	sphereRadius = 0.5

	sunDirection = [0.8, 0.7, 0.5]
	sunDirection = sunDirection / np.linalg.norm(sunDirection)
	skyColor = np.array([138,229,255]) / 255.0
	sunColor = np.array([1.0,1.0,1.0])
	sunStrength = 0.05
	sunAmbientPower = 0.05

	epsilon = 0.01

	farPlane = 15


	specularitySamples = 10


	material = None


	intersectionDepth = -1

	intersectionNormal = None

	#world contains a list of renderables 
	#a renderable is a tuple composed of (geom, material)

	for renderable in world:
		geom, mat = renderable
		candidateDepth, normal = geom.hitsRay(ray, position, epsilon)
		if candidateDepth > epsilon and (candidateDepth < intersectionDepth or intersectionDepth < epsilon):
			intersectionDepth = candidateDepth
			intersectionNormal = normal
			material = mat

	if intersectionDepth < epsilon or intersectionDepth * ray[2] > farPlane:
		return sunColor * sunStrength * (np.dot(ray, sunDirection) / 2.0 + 0.5)**2 + skyColor
	else:
		if bounce == maxBounce:
			return np.array([0,0.0,0])
		else:
			if bounce > 0:
				specularitySamples = 1
			#To do actually add texturing functionality
			color = material.sampleTexture(ray * intersectionDepth)
			incomingColor = np.array([0.0,0.0,0.0])
			for i in xrange(specularitySamples):
				nNorm = intersectionNormal
				if specularitySamples > 1:
					#mod = np.random.normal(0,0.1 * (1 - mat.surfaceSpecularity)**2,3)
					#the second array if a randomized offset
					mod = np.sin(intersectionDepth * ray * mat.surfaceSpecularity + np.array([0.3,125.5,10])) * 0.1 * (1 - mat.surfaceSpecularity)
					nNorm += mod
					nNorm /= np.linalg.norm(nNorm)
				newRayDirection = ray + 2 * np.dot(-ray, nNorm) * nNorm
				newRayDirection /= np.linalg.norm(newRayDirection)
				incomingColor += castRay(world, intersectionDepth * ray, newRayDirection, bounce + 1) / float(specularitySamples)
			shadowColor = castRay(world, intersectionDepth * ray, -sunDirection, maxBounce)
			shadowColor += sunAmbientPower * sunColor
			#shadowColor = 0
			incomingColor = material.surfaceSpecularity * incomingColor
			return (incomingColor + shadowColor) * np.array(color)


##########
# Build the world!
##########

redMat = Material()
redMat.color = [1,0.1,0.1]

blueMat = Material()
blueMat.color = [0.1,0.1,1]

greenMat = Material()
greenMat.color = [0.1,1,0.1]

mirrorMat = Material()
mirrorMat.color = [1,1,1]
mirrorMat.surfaceSpecularity = 0.95

materials = [redMat, blueMat, greenMat, mirrorMat, mirrorMat, mirrorMat]

planeMat = Material()
planeMat.color = [0.4,0.4,0.4]
planeMat.surfaceSpecularity = 0.14

def marble(pos):
	return np.sin(pos[0] * 3 + pos[1] * 7 + pos[2] * 4.5 + noise.snoise3(pos[0] * 12, pos[1] * 12, pos[2] * 12) * 3) * 0.05 + 0.8

def checkerBoard(pos):
	scale = 2
	scaleOverTwo = scale / 2.0
	return float(((pos[0] % scale < scaleOverTwo) and (pos[1] % scale < scaleOverTwo)  and (pos[2] % scale < scaleOverTwo))
	 or (pos[0] % scale > scaleOverTwo) and (pos[1] % scale > scaleOverTwo)  and (pos[2] % scale < scaleOverTwo)
	 or ((pos[0] % scale > scaleOverTwo) and (pos[1] % scale < scaleOverTwo)  and (pos[2] % scale > scaleOverTwo)) 
	 or ((pos[0] % scale < scaleOverTwo) and (pos[1] % scale > scaleOverTwo)  and (pos[2] % scale > scaleOverTwo))) * 0.65 + 0.35

planeMat.sampleTexture = checkerBoard

world = [(Plane(0.7), planeMat)]

for i in xrange(5):
	world.append((Sphere((np.random.rand(3) - 0.5) * 2 + np.array([0,0,0]), 0.2), materials[np.random.randint(6)]))
	world[-1][0].spherePosition[1] = (np.random.rand() - 0.5) - 0.1
	world[-1][0].spherePosition[2] = (np.random.rand() - 0.5) * 0.5 + 4
	print world[-1][1].color

##########
# Cast the rays
##########
for xx in xrange(RESOLUTION[0]):
	print xx
	for yy in xrange(RESOLUTION[1]):
		ray = np.array([np.tan((xx/float(RESOLUTION[0]) - 0.5) * FOV / 2.0), 
			np.tan((yy/float(RESOLUTION[1]) - 0.5) * FOV / 2.0 * 1.0 / aspect), 1])
		ray /= np.linalg.norm(ray)
		screenPixels[yy, xx] = castRay(world, cameraPosition, ray, 0)

screenPixels = np.clip(screenPixels,0,1)
screenPixels *= 255

img = PIL.Image.fromarray(np.uint8(screenPixels))

img.save("renderResult.png")

plt.imshow(img)

plt.show()