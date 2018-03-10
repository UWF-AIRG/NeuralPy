from scipy.io import loadmat
import pickle
import os
import numpy as np

def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros((62, 1), dtype=np.float32)
    e[j] = 1.0
    return e

def load_data(mat_file_path, width=28, height=28, max_=None, verbose=True):
    ''' Load data in from .mat file as specified by the paper.
        Arguments:
            mat_file_path: path to the .mat, should be in sample/
        Optional Arguments:
            width: specified width
            height: specified height
            max_: the max number of samples to load
            verbose: enable verbose printing
        Returns:
            A tuple of training and test data, and the mapping for class code to ascii value,
            in the following format:
                - ((training_images, training_labels), (testing_images, testing_labels), mapping)
    '''
    # Local functions
    def reshape(img):
        # Used to rotate images (for some reason they are transposed on read-in)
        img.shape = (width,height)
        img = img.T
        img = list(img)
        img = [item for sublist in img for item in sublist]
        return img

    def display(img, threshold=0.5):
        # Debugging only
        render = ''
        for row in img:
            for col in row:
                if col > threshold:
                    render += '@'
                else:
                    render += '.'
            render += '\n'
        return render

    # Load convoluted list structure form loadmat
    mat = loadmat(mat_file_path)

    # Load char mapping
    mapping = {kv[0]:kv[1:][0] for kv in mat['dataset'][0][0][2]}
    if not os.path.exists("bin"):
      os.mkdir("bin")
    pickle.dump(mapping, open('bin/mapping.p', 'wb+'))

    # Load training data
    if max_ == None:
        max_ = len(mat['dataset'][0][0][0][0][0][0])
    training_images = mat['dataset'][0][0][0][0][0][0][:max_]
    training_labels = mat['dataset'][0][0][0][0][0][1][:max_]

    # Load testing data
    if max_ == None:
        max_ = len(mat['dataset'][0][0][1][0][0][0])
    else:
        max_ = int(max_ / 6)
    testing_images = mat['dataset'][0][0][1][0][0][0][:max_]
    testing_labels = mat['dataset'][0][0][1][0][0][1][:max_]
    
    # Vectorize classes
    training_labels = [vectorized_result(training_labels[i][0]) for i in range(len(training_labels))]
    testing_labels = [vectorized_result(testing_labels[i][0]) for i in range(len(testing_labels))]

    # Reshape training data to be valid
    if verbose == True: _len = len(training_images)
    for i in range(len(training_images)):
        if verbose == True: print('%d/%d (%.2lf%%)' % (i + 1, _len, ((i + 1)/_len) * 100), end='\r')
        training_images[i] = reshape(training_images[i])
    if verbose == True: print('')

    # Reshape testing data to be valid
    if verbose == True: _len = len(testing_images)
    for i in range(len(testing_images)):
        if verbose == True: print('%d/%d (%.2lf%%)' % (i + 1, _len, ((i + 1)/_len) * 100), end='\r')
        testing_images[i] = reshape(testing_images[i])
    if verbose == True: print('')

    print(display(training_images[0].reshape(height, width, 1), threshold=128))
    print(training_labels[0])
    
    # Flatten, reshape to (width*height, 1), convert type to float32, and normalize pixel values to 0..1
    training_images = [ti.flatten().reshape(height*width, 1).astype('float32') / 255 for ti in training_images]
    testing_images = [ti.flatten().reshape(height*width, 1).astype('float32') / 255 for ti in testing_images]

    nb_classes = len(mapping)

    return (zip(training_images, training_labels), zip(testing_images, testing_labels), mapping, nb_classes)