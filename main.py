import os
import cv2
import numpy as np

from cnn import predict as cnn_predict
from script import get_faces


def RecogniseFace(image=None, feature_type=None, classifier_name=None, creative_mode=0):
    """
    :param image: path to image (relative or absolute)
    :param feature_type: HOG or SURF
    :param classifier_name: CNN or SVM
    :param creative_mode: 0 = False, 1 = True
    :return:
    """

    # Prepare the inputs
    if not (image and classifier_name):
        print("Must provide an image and classifier name")
        return None

    classifier_name = classifier_name.upper()

    if not os.path.isfile(image):
        print("Image file could not be found")
        return None

    if classifier_name not in ['CNN', 'SURF']:
        print("Must provide a supported classified name: CNN or SVM")
        return None

    feature_type = feature_type.upper() if feature_type else ""

    if classifier_name == 'SVM' and feature_type.upper() not in ['HOG', 'SURF']:
        print('Must provide a supported feature type for SVM: HOG or SURF')
        return None

    image_raw = cv2.imread(image)

    # Inputs are fine, lets start producing the output
    if classifier_name == "CNN":
        face_positions = get_faces(image_raw, position=True)

        face_predictions = \
            [ [cnn_predict(image_raw[y:y + h, x:x + w])[0], x + (w/2), y + (h/2)] for (x, y, w, h) in face_positions]

        print(np.matrix(face_predictions))



# if __name__ == "__main__":
#     RecogniseFace()