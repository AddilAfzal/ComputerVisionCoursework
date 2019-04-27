from collections import Counter
from os import listdir

import numpy as np
import cv2

from helper_functions import image_resize, hog, deskew
from settings import face_folders_location


def train_knn(feature_type="SIFT"):

    faces, labels = [], []
    test_faces, test_labels = [], []

    training_faces_location = face_folders_location + "train/"

    folders = listdir(training_faces_location)

    for index, folder in enumerate(folders):
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

        print("%s/%s" % (index + 1, len(folders)))

    print("Training the SVM")

    # Initiate kNN, train the data, then test it with test data for k=1
    knn = cv2.ml.KNearest_create()

    knn.train(np.float32(faces),cv2.ml.ROW_SAMPLE, np.array(labels))
    ret, result, neighbours, dist = knn.findNearest(np.float32(test_faces), k=5)

    test_results_compared = [int(x) == int(y) for x, y in zip(result, test_labels)]

    counter = Counter(test_results_compared)
    t, f = counter[True], counter[False]

    print("Accuracy %s/%s (%s%%)" % (t, t+f, round(t/(t+f)*10000)/100))


def predict_knn(faces, transformed=False):
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

    svm = cv2.ml.SVM_load('trained_data_files/hog_svm.yml')
    return svm.predict(np.float32(descriptors))[1].ravel()


def image_transform(image, feature_type="HOG"):
    """
    Resize the image, then convert to greyscale, then deskew, then compute hog.
    :param image:
    :param feature_type:
    :return: image transformed
    """

    if feature_type == "HOG":
        image = image_resize(image, width=30)
        image_greyscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_greyscale_deskewed = deskew(image_greyscale)
        return hog(image_greyscale_deskewed)

    elif feature_type == "SIFT":
        image = image_resize(image, width=200)
        image_greyscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        bow = cv2.BOWKMeansTrainer(50)

        sift = cv2.xfeatures2d.SIFT_create()
        kp, descriptors = sift.detectAndCompute(image_greyscale, None)

        [bow.add(x) for x in descriptors]

        return bow.cluster()
