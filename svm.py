from os import listdir

import cv2, numpy as np
from cv2.xfeatures2d import SIFT_create

from helper_functions import image_resize
from settings import face_folders_location

"""
Notes
Changed 3 fields to get the best result:
image width, C and Gamma, using trial and error.
https://stackoverflow.com/questions/37715160/how-do-i-train-an-svm-classifier-using-hog-features-in-opencv-3-0-in-python?rq=1
https://docs.opencv.org/trunk/dd/d3b/tutorial_py_svm_opencv.html
"""

SZ=20
bin_n = 16 # Number of bins


def hog(img):
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy)
    bins = np.int32(bin_n*ang/(2*np.pi))    # quantizing binvalues in (0...16)
    bin_cells = bins[:10,:10], bins[10:,:10], bins[:10,10:], bins[10:,10:]
    mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]
    hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)     # hist is a 64 bit vector
    return hist


def deskew(img):
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        # no deskewing needed.
        return img.copy()
    # Calculate skew based on central momemts.
    skew = m['mu11']/m['mu02']
    # Calculate affine transform to correct skewness.
    M = np.float32([[1, skew, -0.5*SZ*skew], [0, 1, 0]])
    # Apply affine transform
    img = cv2.warpAffine(img, M, (SZ, SZ), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return img


def train_svm(feature_type="HOG"):
    """
    Get faces from folders and compute hog for each.
    Each face will be de-skewed first.
    :return:
    """

    faces, labels = [], []
    test_faces, test_labels = [], []

    training_faces_location = face_folders_location + "train/"

    folders = listdir(training_faces_location)

    for folder in folders:
        person_directory = training_faces_location + folder + "/"
        all_filenames = listdir(person_directory)

        sample_filenames = np.random.choice(all_filenames, 10, replace=False) # Extract 10 images from each folder
        # test_filenames = list(set(sample_filenames) - set(all_filenames))

        for filename in all_filenames:
            image = cv2.imread(person_directory + filename)
            image_transformed = image_transform(image, feature_type)

            # If the image was selected at random, add it to the faces/labels list, otherwise add to test set.
            if image_transformed is not None:
                if filename in sample_filenames:
                    faces.append(image_transformed)
                    labels.append(int(folder))
                else:
                    test_faces.append(image_transformed)
                    test_labels.append(int(folder))

            print(folder)

    print("Training the SVM")

    svm = cv2.ml.SVM_create()
    # Set SVM type
    svm.setType(cv2.ml.SVM_C_SVC)
    # Set SVM Kernel to Radial Basis Function (RBF)
    svm.setKernel(cv2.ml.SVM_RBF)
    # # Set parameter C
    # svm.setC(1) # To adjust
    # # Set parameter Gamma
    # svm.setGamma(1) # To adjust
    svm.setC(22)
    svm.setGamma(1.055e-07)

    # Train SVM on training data
    svm.train(np.float32(faces), cv2.ml.ROW_SAMPLE, np.array(labels))
    # svm.trainAuto(np.float32(faces).reshape(-1, 64), cv2.ml.ROW_SAMPLE, np.array(labels))

    # Save trained model
    svm.save("hog_svm.yml")

    results = predict_svm(test_faces, transformed=True)

    q = zip(results, test_labels)

    t, f = 0, 0
    for (x, y) in list(q):
        t += 1 if int(x) == y else 0
        f += 1 if int(x) != y else 0

    print("Accuracy %s/%s (%s%%)" % (t, t+f, round(t/(t+f)*10000)/100))

    return svm


def predict_svm(faces, transformed=False):
    """
    Given a list of faces, predict their labels.
    :param faces:
    :param transformed: whether the list of imaged has already been transformed.
    :return:
    """
    if not transformed:
        descriptors = []

        for image in faces:
            descriptors.append(image_transform(image))
    else:
        descriptors = faces

    svm = cv2.ml.SVM_load('hog_svm.yml')
    return svm.predict(np.float32(descriptors))[1].ravel()


def image_transform(image, feature_type="HOG"):
    """
    Resize the image, then convert to greyscale, then deskew, then compute hog.
    :param image:
    :param feature_type: HOG or SURF
    :return: image transformed
    """

    if feature_type == 'SURF':

        image = image_resize(image, width=150)
        image_greyscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        print(image_greyscale.shape)

        # detector = cv2.FastFeatureDetector_create()
        # kp = detector.detect(image_greyscale, None)
        #
        # br = cv2.BRISK_create()
        # kp, des = br.compute(image_greyscale, kp)
        # return des

        surf = cv2.xfeatures2d.SURF_create()

        kp = surf.detect(image_greyscale)

        # kp = kp if len(kp) == 10 else kp[:10]

        kp, des = surf.compute(image_greyscale, kp)
        return np.float32(des)

        # print(np.float32(des).resize((des.size, 1)))

        return des

    elif feature_type == 'HOG':
        image = image_resize(image, width=30)
        image_greyscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        return hog(deskew(image_greyscale))

    else:
        raise Exception("Missing feature type")
