# NOTE -- please do NOT put your name(s) in the Python code; instead, name the Python file
# itself to include your WPI username(s).

# import cv2  # Uncomment if you have OpenCV and want to run the real-time demo
import numpy as np

def J (w, faces, labels, alpha = 0.):

    y_hats = w.dot(np.transpose(faces))
    y_actuals = labels
    residue = y_hats - y_actuals
    squared = np.square(residue)
    
    return 0.5 * np.sum(squared)


def gradJ (w, faces, labels, alpha = 0.):

    #     Gradient = x^T * (x*w - y)

    gradient = np.transpose(faces).dot((faces.dot(w) - labels))
    return gradient

def gradientDescent (trainingFaces, trainingLabels, testingFaces, testingLabels, alpha = 0.):

    """
    Pick a random starting value for w ∈ R 576 and a small learning rate (epis << 1). Then, using the expression for the gradient 
    of the cost function, iteratively update w to reduce the cost J(w). Stop when the diﬀerence between J over successive training 
    rounds is below some “tolerance” (e.g., δ = 0.001).
    

    
    """
    w = np.zeros(trainingFaces.shape[1])  # Or set to random vect1or
    tolerance = 0.001

    print gradJ(w, trainingFaces, trainingLabels)

    return w

def method1 (trainingFaces, trainingLabels, testingFaces, testingLabels):
    w = np.zeros(trainingFaces.shape[1])  # TODO fix this!

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

    # print(w.size)

    return w

def method2 (trainingFaces, trainingLabels, testingFaces, testingLabels):
    return gradientDescent(trainingFaces, trainingLabels, testingFaces, testingLabels)

def method3 (trainingFaces, trainingLabels, testingFaces, testingLabels):
    alpha = 1e3
    return gradientDescent(trainingFaces, trainingLabels, testingFaces, testingLabels, alpha)

def reportCosts (w, trainingFaces, trainingLabels, testingFaces, testingLabels, alpha = 0.):
    print "Training cost: {}".format(J(w, trainingFaces, trainingLabels, alpha))
    print "Testing cost:  {}".format(J(w, testingFaces, testingLabels, alpha))

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
    faceDetector = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")  # TODO update the path
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

    # print(trainingFaces.shape)
    # print(trainingLabels.shape)

    w1 = method1(trainingFaces, trainingLabels, testingFaces, testingLabels)
    # w2 = method2(trainingFaces, trainingLabels, testingFaces, testingLabels)
    # w3 = method3(trainingFaces, trainingLabels, testingFaces, testingLabels)

    reportCosts(w1, trainingFaces, trainingLabels, testingFaces, testingLabels)

    # for w in [ w1, w2, w3 ]:
    #     reportCosts(w, trainingFaces, trainingLabels, testingFaces, testingLabels)
    
    #detectSmiles(w3)  # Requires OpenCV
