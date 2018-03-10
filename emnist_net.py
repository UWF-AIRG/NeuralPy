import network
import emnist_mat_loader
from PIL import Image
import math

training_data, test_data, mapping, nb_classes = emnist_mat_loader.load_data('C:\\Users\\carso\\Documents\\Programming\\Python\\Neural\\emnist-matlab\\matlab\\emnist-byclass.mat')

training_data = list(training_data)
test_data = list(test_data)

net = net33.Network([784, 300, 150, 62])

evas, tas = [], []
for epoch in range(30):
    training_data_copy = training_data[:]
    test_data_copy = test_data[:]
    evc, eva, tc, ta = net.SGD(training_data_copy, 
	    1,
	    25,
	    0.5,
	    lmbda=5.0,
	    evaluation_data=test_data_copy,
	    monitor_evaluation_accuracy=True,
	    monitor_evaluation_cost=False,
	    monitor_training_accuracy=True,
	    monitor_training_cost=False)
    
    evas.append(eva)
    tas.append(ta)
    
    Fimages = []
    for x in range(62):
        img = Image.new('RGBA', (28, 28))
        pxs = [0.0 for i in range(784)]
        for y in range(784):
            for z in range(300):
                for p in range(150):
                    pxs[y] += net.weights[0][z][y]*net.weights[1][p][z]*net.weights[2][x][p]
        P = [w for w in pxs if w > 0]
        N = [w for w in pxs if w < 0]
        Pmiw = min(P)
        Pmxw = max(P)
        Nmiw = max(N)
        Nmxw = min(N)
        for y in range(784):
            w = pxs[y]
            if w < 0:
                a = int(math.floor(255*(w-Nmiw)/(Nmxw-Nmiw)))
                img.putpixel((y%28,int(math.floor(y/28))),(255,0,0,a))
            elif w > 0:
                a = int(math.floor(255*(w-Pmiw)/(Pmxw-Pmiw)))
                img.putpixel((y%28,int(math.floor(y/28))),(0,0,255,a))
        Fimages.append(img)
        img.save("EMNIST-Output-Heatmaps/E{}/F{}.png".format(epoch, x))