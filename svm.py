import pickle

import cv2, numpy as np
from os import listdir
from cv2.xfeatures2d import SURF_create
from helper_functions import image_resize, hog, deskew, show_accuracy
from settings import face_folders_location

"""
Notes
Changed 3 fields to get the best result:
image width, C and Gamma, using trial and error.
https://stackoverflow.com/questions/37715160/how-do-i-train-an-svm-classifier-using-hog-features-in-opencv-3-0-in-python?rq=1
https://docs.opencv.org/trunk/dd/d3b/tutorial_py_svm_opencv.html
https://answers.opencv.org/question/183596/how-exactly-does-bovw-work-for-python-3-open-cv3/
"""

surf = cv2.xfeatures2d.SURF_create()
flann_params = dict(algorithm=1, trees=5)
matcher = cv2.FlannBasedMatcher(flann_params, {})


def train_svm(feature_type):
    """
    Get faces from folders and compute hog for each.
    Each face will be de-skewed first.
    :return:
    """

    faces, labels, tmp = [], [], []
    test_faces, test_labels = [], []

    training_faces_location = face_folders_location + "train/"

    folders = listdir(training_faces_location)

    bow = cv2.BOWKMeansTrainer(500)
    bow_extract = cv2.BOWImgDescriptorExtractor(surf, matcher)

    for index, folder in enumerate(folders):
        person_directory = training_faces_location + folder + "/"
        all_filenames = listdir(person_directory)
        print(folder)
        sample_filenames = np.random.choice(all_filenames, 29, replace=False)  # Extract 10 images from each folder
        # test_filenames = list(set(sample_filenames) - set(all_filenames))

        for filename in all_filenames:
            image = cv2.imread(person_directory + filename)
            image_transformed = image_transform(image, feature_type)

            if feature_type == "HOG":
                if filename in sample_filenames:
                    print(folder, filename)
                # If the image was selected at random, add it to the faces/labels list, otherwise add to test set.
                if image_transformed is not None:
                    (faces if filename in sample_filenames else test_faces).append(image_transformed)
                    (labels if filename in sample_filenames else test_labels).append(int(folder))

            elif feature_type == "SURF":
                kp, descriptors = surf.detectAndCompute(image_transformed, None)
                bow.add(descriptors)

                tmp.append((folder, image_transformed))

        print("%s/%s" % (index + 1, len(folders)))

    if feature_type == "SURF":
        bow_extract.setVocabulary(bow.cluster())

        for folder in folders:
            items = list(filter(lambda x: x[0] == folder, tmp))
            sample_indexes = np.random.choice(len(items), 29, replace=False)  # Extract 10 images from each folder

            for i, (label, image) in enumerate(items):
                surfkp = surf.detect(image)
                bowsig = bow_extract.compute(image, surfkp)
                (faces if i in sample_indexes else test_faces).extend(bowsig)
                (labels if i in sample_indexes else test_labels).append(int(label))
            print(folder)

    print("Training the SVM")

    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)   # Set SVM type
    svm.setKernel(cv2.ml.SVM_RBF)   # Set SVM Kernel to Radial Basis Function (RBF)

    if feature_type == "HOG":
        svm.setC(22)                # Set SVM C value - was manually determined
        svm.setGamma(1.055e-07)     # Set SVM Gamma value - was manually determined
        svm.train(np.float32(faces), cv2.ml.ROW_SAMPLE, np.array(labels))

    elif feature_type == "SURF":
        svm.setC(11)                # Set SVM C value - was manually determined
        svm.setGamma(11.63)     # Set SVM Gamma value - was manually determined
        svm.train(np.float32(faces), cv2.ml.ROW_SAMPLE, np.array(labels))

        with open('trained_data_files/surf_svm_bow_pickle.pickle', 'wb') as f:
            cluster = bow.cluster()
            print(cluster.__len__())
            pickle.dump(cluster, f)

    # Save trained model
    svm.save("trained_data_files/%s_svm.yml" % feature_type.lower())

    results = predict_svm(test_faces, feature_type, transformed=True)

    predictions_with_actual = zip(results, test_labels)

    show_accuracy(predictions_with_actual)


def predict_svm(faces, feature_type, transformed=False):
    """
    Given a list of faces, predict their labels.
    :param faces:
    :param feature_type:
    :param transformed: whether the list of imaged has already been transformed.
    :return:
    """
    predict_dataset = list()
    train_file = "trained_data_files/%s_svm.yml" % feature_type.lower()
    print(train_file)
    svm = cv2.ml.SVM_load(train_file)

    if feature_type == "HOG":

        if not transformed:
            for image in faces:
                predict_dataset.append(image_transform(image, feature_type))

            return svm.predict(np.float32(predict_dataset))[1].ravel()

        else:
            return svm.predict(np.float32(faces))[1].ravel()

    elif feature_type == "SURF":
        if not transformed:
            bow_extract = cv2.BOWImgDescriptorExtractor(surf, matcher)

            with open('trained_data_files/surf_svm_bow_pickle.pickle', 'rb') as f:
                dictionary = pickle.load(f)

            bow_extract.setVocabulary(dictionary)

            for face in faces:
                image_transformed = image_transform(face, feature_type)
                surf_kp = surf.detect(image_transformed)
                bow_sig = bow_extract.compute(image_transformed, surf_kp)
                predict_dataset.extend(bow_sig)

        else:
            predict_dataset = faces

        return list(map(int, svm.predict(np.float32(predict_dataset))[1].ravel()))

    else:
        raise Exception("Missing feature type")


def image_transform(image, feature_type):
    """
    Resize the image, then convert to greyscale, then deskew, then compute hog.
    :param image:
    :param feature_type: HOG or SURF
    :return: image transformed
    """

    if feature_type == 'HOG':
        image = image_resize(image, width=30)
        image_greyscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return hog(deskew(image_greyscale))

    elif feature_type == "SURF":
        image_greyscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image_greyscale

    else:
        raise Exception("Missing feature type")
