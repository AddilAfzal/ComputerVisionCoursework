from os import listdir, path
import cv2
import numpy as np

from helper_functions import image_resize, hog, deskew
from script import get_faces

"""
http://kdef.se/download-2/7Yri1UsotH.html
Dawel, A., Wright, L., Irons, J., Dumbleton, R., Palermo, R., O’Kearney, R., & McKone, E.
(2017). Perceived emotion genuineness: normative ratings for popular facial expression
stimuli and the development of perceived-as-genuine and perceived-as-fake sets. Behavior
research methods, 49(4), 1539-1562.
"""

dataset_path = "/home/addil/Desktop/KDEF_and_AKDEF/KDEF/"

happy_label = "HA"
sad_label = "SA"
surprised_label = "SU"
angry_label = "AN"
afraid_label = "AF"
neutral_label = "NE"
other = "DI"


def train_expressions_svm():
    """
    The purpose of this method is to extract faces from the dataset, assign label, pre-process and train a svm.
    :return:
    """

    image_labels = []

    folders = listdir(dataset_path)

    # image_size = []

    for folder_name in folders:
        folder_path = dataset_path + folder_name + "/"
        image_filenames = filter(lambda x: 'FL' not in x and 'FR' not in x, listdir(folder_path))

        print(image_filenames)

        folder_contents = []
        skip = False

        for image_filename in image_filenames:
            print(image_filename)
            name = image_filename[2:]

            if happy_label in name or neutral_label in name:
                label = 0
            elif sad_label in name:
                label = 1
            elif surprised_label in name:
                label = 2
            elif angry_label in name:
                label = 3
            elif afraid_label in name or other in name:
                continue
            else:
                raise Exception("Label error")

            image_path = folder_path + image_filename

            # file_size = path.getsize(image_path)
            # print(file_size)
            # image_size.append(file_size)
            # if file_size/1000 < 20:
            #     # Something is wrong with the image. Skip the folder
            #     skip = True
            #     print("Skipping")
            #     raise Exception("Skipping")
            #     continue

            image = cv2.imread(image_path)
            face = get_faces(image, greyscale=False, check_eyes=False, use_custom_scale=False)

            if len(face) > 0:
                face = face[0]
            else:
                skip = True
                break

            # face = get_faces(image, greyscale=False, check_eyes=False)[0]

            image_transformed = image_transform(face)

            folder_contents.append( (image_transformed, label) )

        if not skip:
            image_labels += folder_contents

    dt = np.dtype('object,float')

    sample_tuple_indexes = np.random.choice(len(image_labels), int(image_labels.__len__()*0.7), replace=False)
    sample_tuples = np.take(np.array(image_labels, dtype=dt), sample_tuple_indexes)
    sample_descriptors, sample_labels = list(zip(*sample_tuples))
    sample_labels = list(map(int, sample_labels))

    test_tuple_indexes = list(set(np.arange(0, image_labels.__len__())) - set(sample_tuple_indexes))
    test_tuples = np.take(np.array(image_labels, dtype=dt), test_tuple_indexes)
    test_descriptors, test_labels = list(zip(*test_tuples))
    test_labels = list(map(int, test_labels))

    svm = cv2.ml.SVM_create()
    # Set SVM type
    svm.setType(cv2.ml.SVM_C_SVC)
    # Set SVM Kernel to Radial Basis Function (RBF)
    svm.setKernel(cv2.ml.SVM_RBF)
    svm.setC(13)    # To adjust
    svm.setGamma(2.4e-08)   # To adjust

    # Train SVM on training data
    svm.train(np.float32(sample_descriptors), cv2.ml.ROW_SAMPLE, np.array(sample_labels))
    # svm.trainAuto(np.float32(faces).reshape(-1, 64), cv2.ml.ROW_SAMPLE, np.array(labels))

    # Save trained model
    svm.save("trained_data_files/expressions_svm.yml")

    results = predict_expression_svm(test_descriptors, transformed=True)

    q = zip(results, test_labels)

    t, f = 0, 0
    for (x, y) in list(q):
        t += 1 if int(x) == y else 0
        f += 1 if int(x) != y else 0

    print("Accuracy %s/%s (%s%%)" % (t, t+f, round(t/(t+f)*10000)/100))

    return svm

    # return faces, labels


def check_folders():
    """
    Method to find which folder has a missing image.
    :return:
    """
    folders = listdir(dataset_path)

    for folder_name in folders:
        folder_path = dataset_path + folder_name + "/"
        image_filenames = list(filter(lambda x: 'FL' not in x and 'FR' not in x, listdir(folder_path)))
        print(image_filenames)
        sad_filenames = list(filter(lambda x: sad_label in x, image_filenames))

        if len(sad_filenames) != 3:
            raise Exception("Folder %s has a missing image" % folder_name)


def predict_expression_svm(faces, transformed=False):
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

    svm = cv2.ml.SVM_load('trained_data_files/expressions_svm.yml')
    return svm.predict(np.float32(descriptors))[1].ravel()


def image_transform(image):
    """
    Resize the image, then convert to greyscale, then deskew, then compute hog.
    :param image:
    :return: image transformed
    """

    image = image_resize(image, width=25)
    image_greyscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return hog(deskew(image_greyscale))
