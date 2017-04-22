import network
import mnist_loader
from visual import *
from visual.graph import *
from PIL import Image

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

print(test_data[0])

net = network.Network([784, 30, 10])

evc, eva, tc, ta = net.SGD(training_data, 
	30,
	10,
	3.0,
	lmbda=5.0,
	evaluation_data=validation_data,
	monitor_evaluation_accuracy=True,
	monitor_evaluation_cost=False,
	monitor_training_accuracy=True,
	monitor_training_cost=False)

f = gcurve(color=color.blue)
f2 = gcurve(color=color.red)
f.plot(pos=(0,0))
f2.plot(pos=(0,0))
for x in xrange(len(eva)):
    f.plot(pos=(x+1,eva[x]/10000.0))
    f2.plot(pos=(x+1,ta[x]/50000.0))

Himages = []
for x in xrange(30):
    img = Image.new('RGBA', (28, 28))
    P = [w for w in net.weights[0][x] if w > 0]
    N = [w for w in net.weights[0][x] if w < 0]
    Pmiw = min(P)
    Pmxw = max(P)
    Nmiw = max(N)
    Nmxw = min(N)
    for y in xrange(784):
        w = net.weights[0][x][y]
        if w < 0:
            a = int(floor(255*(w-Nmiw)/(Nmxw-Nmiw)))
            img.putpixel((y%28,int(floor(y/28))),(255,0,0,a))
        elif w > 0:
            a = int(floor(255*(w-Pmiw)/(Pmxw-Pmiw)))
            img.putpixel((y%28,int(floor(y/28))),(0,0,255,a))
    Himages.append(img)
    img.save("{}.png".format(x))

Fimages = []
for x in xrange(10):
    img = Image.new('RGBA', (28, 28))
    pxs = [0.0 for i in xrange(784)]
    for y in xrange(784):
        for z in xrange(30):
            pxs[y] += net.weights[0][z][y]*net.weights[1][x][z]
    P = [w for w in pxs if w > 0]
    N = [w for w in pxs if w < 0]
    Pmiw = min(P)
    Pmxw = max(P)
    Nmiw = max(N)
    Nmxw = min(N)
    for y in xrange(784):
        w = pxs[y]
        if w < 0:
            a = int(floor(255*(w-Nmiw)/(Nmxw-Nmiw)))
            img.putpixel((y%28,int(floor(y/28))),(255,0,0,a))
        elif w > 0:
            a = int(floor(255*(w-Pmiw)/(Pmxw-Pmiw)))
            img.putpixel((y%28,int(floor(y/28))),(0,0,255,a))
    Fimages.append(img)
    img.save("F{}.png".format(x))
