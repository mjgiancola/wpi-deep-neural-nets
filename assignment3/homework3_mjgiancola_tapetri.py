# NOTE -- please do NOT put your name(s) in the Python code; instead, name the Python file
# itself to include your WPI username(s).

import cv2  # Uncomment if you have OpenCV and want to run the real-time demo
import numpy as np
from scipy.optimize import check_grad

def sigmoid_array(x):
  return 1.0 / (1.0 + np.exp(-1.0 * x))

def J (w, faces, labels, alpha = 0.): # TODO Change to cross-entropy loss function

    y_hats = w.dot(np.transpose(faces))
    y_actuals = labels
    residue = y_hats - y_actuals
    squared = np.square(residue)
    
    reg = 0.5*alpha * w.dot(w)

    return 0.5 * np.sum(squared) + reg

def J_new (w, faces, labels, alpha = 0.): # TODO Change to cross-entropy loss function
    

    y_hats = sigmoid_array(w.dot(np.transpose(faces)))
    print (y_hats)
    y_actuals = labels
    m = y_hats.shape[0]
    ones = np.ones(y_actuals.shape)

    positives = y_actuals*np.log(y_hats) 
    negatives = (ones - y_actuals)*np.log(ones-y_hats)

    regul = 0.5*alpha*np.dot(w.T,w)

    return -1.0/m * np.sum(positives + negatives) + regul

def gradJ_new (w, faces, labels, alpha = 0.):


    y_hats = sigmoid_array(w.dot(np.transpose(faces)))
    y_actuals = labels
    m = y_hats.shape[0]
    print(m)
    result = 1.0/m * np.sum(np.dot(faces.T,np.transpose((y_hats - y_actuals))))
    # result = -1.0/m * np.sum(y_actuals - y_hats)
    return result

def gradJ (w, faces, labels, alpha = 0.): # TODO Change to gradient of cross-entropy loss function
    #     Gradient = x^T * (x*w - y)
    gradient = faces.T.dot(faces.dot(w) - labels)

    return gradient

def gradientDescent (trainingFaces, trainingLabels, testingFaces, testingLabels, alpha = 0.):

    """
    Pick a random starting value for w in R^576 and a small learning rate
    (epis << 1). Then, using the expression for the gradient of the cost
    function, iteratively update w to reduce the cost of J(w). Stop when
    the difference between J over successive training rounds is below some
    tolerance (eg. delta = 0.001).
    """
    learning_rate = 3.4e-6
    tolerance = 1e-3

    w = np.zeros(trainingFaces.shape[1])  # Or set to random vector

    lastJ = np.inf
    currentJ =  J(w, trainingFaces, trainingLabels, alpha) 
    delta = lastJ - currentJ

    while (delta > tolerance):

        print('J = ' + str(currentJ) + ' ||w|| = ' + str(np.linalg.norm(w)))

        lastJ = currentJ
        w = w - ( learning_rate * gradJ(w, trainingFaces, trainingLabels) )
        currentJ = J(w, trainingFaces, trainingLabels, alpha)
        delta = abs(lastJ - currentJ)

        print('Delta = ' + str(delta))

    return w

def method1 (trainingFaces, trainingLabels, testingFaces, testingLabels):
    w = np.zeros(trainingFaces.shape[1])

    """
    gradient = 0
    gives us:
    w = (x^T*X)^-1 * x^Ty
    w = A^(-1) * b
    w = Solve(A,b)

    """
    A = np.transpose(trainingFaces).dot(trainingFaces)
    b = np.transpose(trainingFaces).dot(trainingLabels)
    w = np.linalg.solve(A, b)

    return w

def method2 (trainingFaces, trainingLabels, testingFaces, testingLabels):
    return gradientDescent(trainingFaces, trainingLabels, testingFaces, testingLabels)

def method3 (trainingFaces, trainingLabels, testingFaces, testingLabels):
    alpha = 1e3
    return gradientDescent(trainingFaces, trainingLabels, testingFaces, testingLabels, alpha)

def reportCosts (w, trainingFaces, trainingLabels, testingFaces, testingLabels, alpha = 0.):
    print "Training cost: {}".format(J(w, trainingFaces, trainingLabels, alpha))
    print "Testing cost:  {}".format(J(w, testingFaces, testingLabels, alpha))


def whiten (trainingFaces):
    # cov = np.dot(trainingFaces.T, trainingFaces)
    """ 
    Return matrix L such that (trainingFaces*L)^T(trainingFaces*L) = I
    """
    # w - eigenvalues
    # v - eigenvectors
    # trainingFaces -= np.mean(trainingFaces, axis = 0)
    # cov = np.dot(trainingFaces.T, trainingFaces) # / trainingFaces.shape[0]

    # U,S,V = np.linalg.svd(cov)

    # w = np.sqrt(S + 1e-5)
    # v = U

    v,w = np.linalg.eig(cov)

    lamd = np.diag(np.float_power(w, (-0.5)))
    phi = v

    L = phi.dot(lamd)
    return L

# Accesses the web camera, displays a window showing the face, and classifies smiles in real time
# Requires OpenCV.
def detectSmiles (w):
    # Given the image captured from the web camera, classify the smile
    def classifySmile (im, imGray, faceBox, w):
        # Extract face patch as vector
        face = imGray[faceBox[1]:faceBox[1]+faceBox[3], faceBox[0]:faceBox[0]+faceBox[2]]
        face = cv2.resize(face, (24, 24))
        face = (face - np.mean(face)) / np.std(face)  # Normalize
        face = np.reshape(face, face.shape[0]*face.shape[1])

        # Classify face patch
        yhat = w.dot(face)
        print yhat

        # Draw result as colored rectangle
        THICKNESS = 3
        green = 128 + (yhat - 0.5) * 255
        color = (0, green, 255 - green)
        pt1 = (faceBox[0], faceBox[1])
        pt2 = (faceBox[0]+faceBox[2], faceBox[1]+faceBox[3])
        cv2.rectangle(im, pt1, pt2, color, THICKNESS)

    # Starting video capture
    vc = cv2.VideoCapture()
    vc.open(0)
    faceDetector = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
    while vc.grab():
        (tf,im) = vc.read()
        im = cv2.resize(im, (im.shape[1]/2, im.shape[0]/2))  # Divide resolution by 2 for speed
        imGray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        k = cv2.waitKey(30)
        if k >= 0 and chr(k) == 'q':
            print "quitting"
            break

        # Detect faces
        faceBoxes = faceDetector.detectMultiScale(imGray)
        for faceBox in faceBoxes:
            classifySmile(im, imGray, faceBox, w)
        cv2.imshow("WebCam", im)

    cv2.destroyWindow("WebCam")
    vc.release()

if __name__ == "__main__":
    # Load data
    if ('trainingFaces' not in globals()):  # In ipython, use "run -i homework2_template.py" to avoid re-loading of data
        trainingFaces = np.load("trainingFaces.npy")
        trainingLabels = np.load("trainingLabels.npy")
        testingFaces = np.load("testingFaces.npy")
        testingLabels = np.load("testingLabels.npy")

    w = 1e3 * np.random.choice(5, 576) 
    # w = np.zeros(trainingFaces.shape[1])
    # w = trainingFaces[4]
    # print(w.shape)
    # print(w)
    print (check_grad(J_new, gradJ_new, w, trainingFaces, trainingLabels))


    def func(x):
        return x[0]**2 - 0.5 * x[1]**3
    def grad(x):
        return [2 * x[0], -1.5 * x[1]**2]
    print(check_grad(func, grad, [0,0]))

    # trainingFaces = trainingFaces.dot(whiten(trainingFaces))
    
    # TODO Whiten trainingFaces, testingFaces before running gradient descent
    
    # transformed = trainingFaces.dot(whiten(trainingFaces))
    # cov2 = np.dot(transformed.T, transformed)

    # w, v = np.linalg.eig(cov2)
    # print (w)

    # w1 = method1(trainingFaces, trainingLabels, testingFaces, testingLabels)
    #w2 = method2(trainingFaces, trainingLabels, testingFaces, testingLabels)
    # w3 = method3(trainingFaces, trainingLabels, testingFaces, testingLabels)

    # for w in [ w1, w2, w3 ]:
    # reportCosts(w1, trainingFaces, trainingLabels, testingFaces, testingLabels)
    # reportCosts(w2, trainingFaces, trainingLabels, testingFaces, testingLabels)
    # reportCosts(w3, trainingFaces, trainingLabels, testingFaces, testingLabels)
    
    # detectSmiles(w3)  # Requires OpenCV
