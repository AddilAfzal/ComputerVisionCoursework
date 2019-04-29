import os
import cv2
import face_recognition
import numpy as np

from cnn import predict_cnn
from facial_expressions import predict_expression_svm
from knn import predict_knn
from script import get_faces
from svm import predict_svm


def RecogniseFace(image="", feature_type=None, classifier_name=None):
    """
    :param image: path to image (relative or absolute)
    :param feature_type: HOG or SURF
    :param classifier_name: CNN or SVM
    :param creative_mode: 0 = False, 1 = True
    :return:
    """

    # Prepare the inputs
    if image is None or classifier_name is None:
        print("Must provide an image and classifier name")
        return None

    classifier_name = classifier_name.upper()

    if not os.path.isfile(image):
        print("Image file could not be found")
        return None

    if classifier_name not in ['CNN', 'SVM', 'KNN']:
        print("Must provide a supported classified name: CNN or SVM or KNN")
        return None

    feature_type = feature_type.upper() if feature_type else ""

    if classifier_name == 'SVM' and feature_type.upper() not in ['HOG', 'SURF']:
        print('Must provide a supported feature type for SVM: HOG or SURF')
        return None

    if classifier_name == 'KNN' and feature_type.upper() not in ['HOG', 'SURF']:
        print('Must provide a supported feature type for KNN: HOG or SURF')
        return None

    image_raw = cv2.imread(image)

    # Inputs are fine, lets start producing the output
    if classifier_name == "CNN":
        face_positions = face_recognition.face_locations(image_raw)

        face_predictions = []

        for (top, right, bottom, left) in face_positions:
            prediction = int(predict_cnn(image_raw[top:bottom, left:right])[0])
            emo = int(predict_expression_svm([image_raw[top:bottom, left:right]])[0])
            x,y = int((left+right)/2), int((top+bottom)/2)
            face_predictions.append((prediction, x, y, emo))

        return face_predictions

    elif classifier_name == "SVM":
        face_positions = face_recognition.face_locations(image_raw)

        results = predict_svm([image_raw[top:bottom, left:right] for (top, right, bottom, left) in face_positions],
                              feature_type)

        emo_results = predict_expression_svm([image_raw[top:bottom, left:right] for (top, right, bottom, left) in face_positions])

        results_positions = list(
            [(int(result), int((left + right) / 2), int((top + bottom) / 2), int(emo)) for result, (top, right, bottom, left), emo in
             zip(results, face_positions, emo_results)])

        return results_positions

    elif classifier_name == "KNN":
        face_positions = face_recognition.face_locations(image_raw)

        results = predict_knn([image_raw[top:bottom, left:right] for (top, right, bottom, left) in face_positions], feature_type)

        emo_results = predict_expression_svm([image_raw[top:bottom, left:right] for (top, right, bottom, left) in face_positions])

        results_positions = list(
            [(int(result), int((left + right) / 2), int((top + bottom) / 2), int(emo)) for result, (top, right, bottom, left), emo in
             zip(results, face_positions, emo_results)])

        return results_positions
