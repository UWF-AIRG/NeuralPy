import net3
from visual import *
from visual.graph import *
from PIL import Image
import numpy as np
import random as random

flux = 0.25
r = 3.0
R = 5.0

def datapoint(t):
	theta = 2*math.pi*random.random()
	rplus = flux*(random.random()-random.random())
	if t:
		vx = (R+rplus)*math.cos(theta)
		vy = (R+rplus)*math.sin(theta)
		sx = vx**2
		sy = vy**2
		return [((array([[vx], [vy], [sx], [sy]],dtype=float32),array([[1.0]],dtype=float32)))]
	else:
		tr = (r+rplus)*random.random()
		vx = (tr)*math.cos(theta)
		vy = (tr)*math.sin(theta)
		sx = vx**2
		sy = vy**2
		return [((array([[vx], [vy], [sx], [sy]],dtype=float32),array([[0.0]],dtype=float32)))]

data = []

for i in xrange(1000):
	data += datapoint(random.random() > 0.5)

training_data = data
validation_data = data
test_data = data

net = net3.Network([4, 3, 3, 1])

w = 500
h = 500
for e in xrange(10):
	eva, ta = net.SGD(training_data, 
		1,
		10,
		1.0,
		lmbda=0.001,
		evaluation_data=validation_data,
		monitor_evaluation_cost=True,
		monitor_training_cost=False)

	print "Creating image"
	img = Image.new('RGBA', (w, h))
	pxs = [0.0 for i in xrange(w*h)]
	for xi in xrange(w):
		for yi in xrange(h):
			vx = float(R+r+flux)*float(xi-(w/2))/float(w/2)
			vy = float(R+r+flux)*float(yi-(h/2))/float(h/2)
			d = array([[vx],[vy],[vx**2],[vy**2]],dtype=float32)
			v = net.feedforward(d)[0][0]
			if ((xi*h)+yi)%100000 == 0: print float((xi*h)+yi) /float(w*h)
			alpha = 255
			red = int(((0.5-v)/0.5)*255.0) if v < 0.5 else 0
			green = 0
			blue = int(((v-0.5)/0.5)*255.0) if v > 0.5 else 0
			img.putpixel((xi,yi),(red,green,blue,alpha))
	img.save("train-{}.png".format(e))
	print "Created image"