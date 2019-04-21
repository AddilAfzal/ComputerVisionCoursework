from shutil import rmtree
from os import listdir, path, makedirs

import cv2

face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('cascades/haarcascade_eye.xml')
profile_face_cascade = cv2.CascadeClassifier('cascades/haarcascade_profileface.xml')


def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    """
    https://stackoverflow.com/questions/44650888/resize-an-image-without-distortion-opencv
    :param image:
    :param width:
    :param height:
    :param inter:
    :return:
    """
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized


def get_sharpness(img):
    """
    Compute a sharpness value for a given image.
    :param img: Greyscale image
    :return:
    """
    return cv2.Laplacian(img, cv2.CV_64F).var()


def number_of_eyes(img):
    """
    Return the number of eyes found within an image.
    :param img: grey scale image
    :return:
    """
    return eye_cascade.detectMultiScale(img).__len__()


def reset_folders(only=[], full=False):
    """
    Remove the training images from each folder.
    :param only: Define a list of folder ids to reset.
    :param full: True will also remove the angle folders.
    :return:
    """

    from script import folders_location

    only = list(map(str, only))

    individuals = only if only else listdir(folders_location)

    for individual_folder_name in individuals:

        individual_folder_path = folders_location + individual_folder_name + "/"

        individual_training_folder = individual_folder_path + "training"
        if path.exists(individual_training_folder):
            rmtree(individual_training_folder)

        if full:
            for f in listdir(individual_folder_path):
                if f.__contains__("angle"):
                    rmtree(individual_folder_path + f)


def get_image_sizes():
    """
    Return a list of image sized that were added to the training folder.
    :return:
    """
    widths = []
    heights = []

    from script import folders_location
    for individual_folder_name in listdir(folders_location):
        individual_training_folder_path = folders_location + individual_folder_name + "/training/"

        image_paths = listdir(individual_training_folder_path)
        for image_path in image_paths:
            img = cv2.imread(individual_training_folder_path + image_path)

            height, width, channel = img.shape
            widths.append(width)
            heights.append(height)

            print(individual_training_folder_path + image_path)

    print("Min: %s, Max: %s" % (np.min(widths), np.max(widths)))
    print("Average: %s" % (np.average(widths)))

    return widths