from operator import itemgetter

import face_recognition
from math import floor
from sklearn.metrics import confusion_matrix

from cnn import predict_cnn, start_training
from facial_expression_cnn import predict_expression
from helper_functions import *
import cv2, subprocess as sp, numpy as np

from knn import predict_knn
from settings import folders_location, cnn_folders_location, face_folders_location, group_images_location, \
    group_image_faces_location, group_image_faces_location_two, group_videos_location, group_video_images_location, \
    group_video_faces_location, group_video_faces_location_2, facial_expressions_folder
from svm import predict_svm

recognizer = cv2.face.LBPHFaceRecognizer_create()


def show_image(img):

    cv2.imshow('img', img)
    cv2.waitKey()
    cv2.destroyAllWindows()


def get_faces(img, position=False, greyscale=True, check_eyes=False, use_custom_scale=True):
    """
    Given an image, extract all faces.
    :param use_custom_scale:
    :param check_eyes:
    :param greyscale:
    :param img: image, not path
    :param position: Whether to return the position or raw image.
    :return: A list of faces in grey
    """

    # Convert the image to greyscale
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    scale = [1.1, 25] if use_custom_scale else []

    # Find the faces
    faces = face_cascade.detectMultiScale(grey, *scale)
    profile_faces = profile_face_cascade.detectMultiScale(grey, *scale)

    # Where the faces will be stored once extracted.
    faces_list = []

    # For each frontal face detected.
    for (x, y, w, h) in faces:

        # Extract the face from the image.
        face = grey[y:y + h, x:x + w] if greyscale else img[y:y + h, x:x + w]

        # Check if this contains eyes. This is needed to prevent noise as being found as a face.
        if not check_eyes or number_of_eyes(face) in [1, 2]:

            # Add to the list of faces.
            faces_list.append((x, y, w, h) if position else face)

    # If there weren't any frontal faces in the image, we'll look for profile faces
    # For each profile face detected
    for (x, y, w, h) in profile_faces:
        face = grey[y:y + h, x:x + w] if greyscale else img[y:y + h, x:x + w]

        if not check_eyes or number_of_eyes(face) in [1, 2]:
            faces_list.append((x, y, w, h) if position else face)

    # if not faces_list:
    #     print("No face")

    # if len(faces_list) == 0 and check_eyes is True:
    #     return get_faces(img=img, position=position, greyscale=greyscale, check_eyes=False,
    #                      use_custom_scale=use_custom_scale)
    # else:
    return faces_list


def get_faces_with_sharpness(angle_folder_path):
    faces_list = []

    # Find faces in each of the images, compute sharpness and store in list.
    j = 0
    for index, file_name in enumerate(listdir(angle_folder_path)):
        image_file_path = angle_folder_path + file_name
        img = cv2.imread(image_file_path)
        print(image_file_path)

        for face in get_faces(img):
            faces_list.append({
                'sharpness': get_sharpness(face),
                'image': face,
            })
            print(j)
            j += 1

    return faces_list


def sort_and_save(faces_list, individual_training_folder, angle_folder_name, n):
    """
    Given a list of faces, sort by sharpness rating and then save the n sharpest.
    :param faces_list:
    :param individual_training_folder:
    :param angle_folder_name:
    :param n:
    :return:
    """

    sorted_faces = sorted(faces_list, key=itemgetter('sharpness'), reverse=True)
    q = 0
    for obj in sorted_faces[:n]:
        q += 1
        cv2.imwrite("%s/%s-%s.jpg" % (individual_training_folder, angle_folder_name, q), obj['image'])
        del obj['image']


def grab_additional_images(only=[], num_images=3):
    """
    Function to grab an additional n number of 'sharpest' images from each video file/angle.
    :param training_path:
    :param num_images:
    :return:
    """

    only = list(map(str, only))
    individuals = only if only else listdir(folders_location)

    for individual in individuals:
        individual_folder_path = folders_location + individual + "/"

        if path.isdir(individual_folder_path):

            # Create the training folder if it doesn't already exist.
            individual_training_folder = individual_folder_path + "training/"
            if not path.exists(individual_training_folder):
                makedirs(individual_training_folder)

            # For each folder of pictures starting with the name 'angle', grab the n sharpest images.
            for angle_folder_name in list(filter(lambda i: i.__contains__("angle"), listdir(individual_folder_path))):
                angle_folder_path = individual_folder_path + angle_folder_name + "/"

                faces_list = get_faces_with_sharpness(angle_folder_path)

                sort_and_save(faces_list, individual_training_folder, angle_folder_name, num_images)

                del faces_list[:]
                del faces_list


def convert_videos_to_images(only=[]):
    """
    For each folder, extract images from each of the videos
    :param only:
    :return:
    """
    only = list(map(str, only))
    individuals = only if only else listdir(folders_location)

    for individual_folder_name in individuals:
        individual_folder_path = folders_location + individual_folder_name + "/"

        if path.isdir(individual_folder_path):

            q = 1
            video_file_names = list(filter(lambda i: i[-3:] in ['mov', 'mp4'], listdir(individual_folder_path)))
            for video_file_name in video_file_names:

                # Folder per video file.
                image_folder_path = individual_folder_path + ("angle_%s/" % q)
                # print(video_folder_path)

                if not path.exists(image_folder_path):
                    makedirs(image_folder_path)

                video_file_path = individual_folder_path + video_file_name

                print(video_file_path)

                cmd = 'ffmpeg -i \'' + video_file_path + '\' -qscale:v 2 -r 10.0 \'' + image_folder_path + '_%03d.jpg\''
                sp.call(cmd, shell=True)

                q += 1


def prepare_dataset(only=[], resize=True):
    only = list(map(str, only))

    individuals = only if only else listdir(folders_location)

    faces = []
    labels = []

    for individual_folder_name in individuals:
        individual_training_folder_path = folders_location + individual_folder_name + "/training/"

        image_paths = listdir(individual_training_folder_path)
        for image_path in image_paths:
            img = cv2.imread(individual_training_folder_path + image_path)
            grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            faces.append(image_resize(grey, height=100) if resize else grey)
            labels.append(int(individual_folder_name))

            print(individual_training_folder_path + image_path)

    return faces, np.array(labels)


def recognise_face(img_path=None, img=None):

    if img_path:
        img = cv2.imread(img_path)

    recognizer.read('trained')

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    id, accuracy = recognizer.predict(gray)

    return id, accuracy


def find_faces_and_label():
    img = cv2.imread('images/group.jpg')

    faces = face_recognition.face_locations(img)
    # faces = get_faces(img, position=True, greyscale=False, check_eyes=False)

    tmp = [img[top:bottom, left:right] for top, right, bottom, left in faces]

    predictions = predict_cnn(images=tmp)
    from facial_expressions import predict_expression_svm
    predictions_expressions = predict_expression_svm(tmp)

    for face, prediction, prediction_exp in zip(faces, predictions, predictions_expressions):
        top, right, bottom, left = face

        label = prediction_exp

        cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 3)
        cv2.putText(img, "%s" % int(prediction), (left + 3, bottom - 4), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 170, 0), lineType=cv2.LINE_AA)

    cv2.imwrite('output.jpg', img)
    cv2.imshow('img', cv2.resize(img, None, fx=0.5, fy=0.5))
    cv2.waitKey()
    cv2.destroyAllWindows()


def format_cnn(only=[]):
    """
    For each folder, extract images from each of the videos
    :return:
    """
    individuals = only if only else listdir(cnn_folders_location)

    for individual_folder_name in individuals:
        individual_folder_path = folders_location + individual_folder_name + "/"
        cnn_individual_folder_path = cnn_folders_location + individual_folder_name + "/"

        if path.isdir(individual_folder_path):

            q = 1
            video_file_names = list(filter(lambda i: i[-3:] in ['mov', 'mp4'], listdir(individual_folder_path)))
            for video_file_name in video_file_names:

                # Folder per video file.
                cnn_image_folder_path = cnn_individual_folder_path

                if not path.exists(cnn_individual_folder_path):
                    makedirs(cnn_individual_folder_path)

                video_file_path = individual_folder_path + video_file_name

                print(video_file_path)

                cmd = ('ffmpeg -i \'' + video_file_path + '\' -qscale:v 2 -r 20.0 \'' + cnn_image_folder_path + str(q) + '_%03d-' + '.jpg\'')
                sp.call(cmd, shell=True)

                q += 1


def create_test_set():
    val_percentage = 0.3
    test_percentage = 0.1

    individuals = listdir(face_folders_location + "train/")

    for individual_folder_name in individuals:
        individual_folder_path = face_folders_location + "train/" + individual_folder_name + "/"
        val_individual_folder_path = face_folders_location + "val/" + individual_folder_name + "/"
        test_individual_folder_path = face_folders_location + "test/" + individual_folder_name + "/"

        if not path.exists(val_individual_folder_path):
            makedirs(val_individual_folder_path)

        if not path.exists(test_individual_folder_path):
            makedirs(test_individual_folder_path)

        image_filenames = listdir(individual_folder_path)
        num_of_images = len(image_filenames)

        # Sample a number of images to take from the folder.
        sample_images = np.random.choice(image_filenames, floor(num_of_images*val_percentage), replace=False)
        remaining_image = list(set(image_filenames) - set(sample_images))
        test_images = np.random.choice(remaining_image, floor(num_of_images*test_percentage), replace=False)

        print("%s of %s" % (sample_images.__len__(), num_of_images))

        for sample_image in sample_images:
            cmd = "mv '%s' '%s'" % (individual_folder_path + sample_image, val_individual_folder_path + sample_image)
            sp.call(cmd, shell=True)

        for test_image in test_images:
            cmd = "mv '%s' '%s'" % (individual_folder_path + test_image, test_individual_folder_path + test_image)
            sp.call(cmd, shell=True)

        print(individual_folder_path)


def extract_faces(only=[]):

    individuals = only if only else listdir(cnn_folders_location)

    for individual_folder_name in individuals:
        individual_folder_path = cnn_folders_location + individual_folder_name + "/"
        face_individual_folder_path = face_folders_location + "train/" + individual_folder_name + "/"

        if not path.exists(face_individual_folder_path):
            makedirs(face_individual_folder_path)

        if path.isdir(individual_folder_path):
            image_filenames = listdir(individual_folder_path)

            i = 1

            for image_filename in image_filenames:
                image_file_path = individual_folder_path + image_filename
                # print(image_file_path)
                faces_list = get_faces(cv2.imread(image_file_path), greyscale=False)

                if faces_list:
                    save_path = face_individual_folder_path + str(i) + ".jpg"
                    print("saving to %s" % save_path)
                    cv2.imwrite(save_path, faces_list[0])
                    i += 1


def extract_faces_from_groups():
    group_images = list(filter(lambda i: i[-3:] in ['jpg'], listdir(group_images_location)))
    print(group_images)

    i = 1
    for group_image_name in group_images:
        image = cv2.imread(group_images_location + group_image_name)
        faces = get_faces(image, greyscale=False)

        for face in faces:
            p1, p2 = predict_cnn(face)

            cv2.imwrite('%s%s-%s.jpg' % (group_image_faces_location, p1, str(i)), face)
            i += 1


def combine_group_extract_with_cnn():
    individuals = listdir(folders_location)

    for individual in individuals:
    #     group_extracts = list(filter(lambda i: i[-3:] in ['jpg'], listdir(
    #         folders_location + individual + "/group_extract")))

        group_extracts_2 = list(filter(lambda i: i[-3:] in ['jpg'], listdir(
            folders_location + individual + "/group_extract_2")))

        # for image_name in group_extracts:
        #     cmd = "cp '%s' '%s'" % (folders_location + individual + "/group_extract/" + image_name, face_folders_location + "/train/" + individual + "/")
        #     sp.call(cmd, shell=True)

        for image_name in group_extracts_2:
            cmd = "cp '%s' '%s'" % (folders_location + individual + "/group_extract_2/" + image_name, face_folders_location + "/train/" + individual + "/")
            sp.call(cmd, shell=True)


def re_label_group_extractions():
    files = list(filter(lambda i: i[-3:] in ['jpg'], listdir(group_image_faces_location)))
    print(files)

    i = 1
    for file_name in files:
        face_image = cv2.imread(group_image_faces_location + file_name)

        p1, p2 = predict(face_image)

        cv2.imwrite('%s%s-%s.jpg' % (group_image_faces_location_two, p1, str(i)), face_image)
        i += 1


def prepare_cnn():
    extract_faces()
    combine_group_extract_with_cnn()
    create_test_set()
    start_training()
    re_label_group_extractions()


def extract_frames_from_group_videos():

    video_file_names = list(filter(lambda i: i[-3:] in ['mov', 'mp4'], listdir(group_videos_location)))

    q = 0

    for video_file_name in video_file_names:

        # Folder per video file.
        image_folder_path = group_video_images_location + ("angle_%s/" % q)
        # print(video_folder_path)

        if not path.exists(image_folder_path):
            makedirs(image_folder_path)

        video_file_path = group_videos_location + video_file_name

        print(video_file_path)

        cmd = 'ffmpeg -i \'' + video_file_path + '\' -qscale:v 2 \'' + image_folder_path + '_%03d.jpg\''
        sp.call(cmd, shell=True)

        q += 1


def extract_faces_from_frames_and_label():

    angles = listdir(group_video_images_location)
    i = 0

    for angle in angles:
        angle_folder_location = group_video_images_location + angle + "/"
        images = listdir(angle_folder_location)
        print(angle_folder_location)

        for index, image_filename in enumerate(images):

            group_image = cv2.imread(angle_folder_location + image_filename)
            faces = get_faces(group_image, greyscale=False)

            for face in faces:
                p1, acc = predict(face)

                cv2.imwrite('%s%s-%s.jpg' % (group_video_faces_location, p1, str(i)), face)
                i += 1


def re_label_group_extractions_2():
    files = list(filter(lambda i: i[-3:] in ['jpg'], listdir(group_video_faces_location)))
    print(files)

    i = 1
    for file_name in files:
        face_image = cv2.imread(group_video_faces_location + file_name)

        p1, p2 = predict(face_image)

        cv2.imwrite('%s%s-%s.jpg' % (group_video_faces_location_2, p1, str(i)), face_image)
        i += 1


def display_matrix(matrix, img):
    img = cv2.imread(img)

    for i, x, y in matrix:
        print(i, x, y)
        cv2.putText(img, str(i), (x, y), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 170, 0), lineType=cv2.LINE_AA)

    cv2.imshow('img', img)
    cv2.waitKey()
    cv2.destroyAllWindows()


def sort_facial_expressions():
    dataset_path = "/home/addil/Desktop/KDEF_and_AKDEF/KDEF/"

    happy_label = ("HA", "Happy")
    sad_label = ("SA", "Sad")
    surprised_label = ("SU", "Surprised")
    angry_label = ("AN", "Angry")
    afraid_label = ("AF", "Afraid")
    neutral_label = ("NE", "Neutral")
    other = ("DI", "DI?")

    folders = listdir(dataset_path)

    for folder_name in folders:
        folder_path = dataset_path + folder_name + "/"
        image_filenames = filter(lambda x: 'FL' not in x and 'FR' not in x, listdir(folder_path))

        print(image_filenames)

        for image_filename in image_filenames:
            image_path = folder_path + image_filename
            image = cv2.imread(image_path)

            face = get_faces(image, greyscale=False, check_eyes=True, use_custom_scale=False)

            if len(face) > 0:
                face = face[0]
            else:
                continue

            print(image_filename)

            filename_striped = image_filename[2:]

            if happy_label[0] in filename_striped:
                label = happy_label[1]
            elif sad_label[0] in filename_striped:
                label = sad_label[1]
            elif surprised_label[0] in filename_striped:
                label = surprised_label[1]
            elif angry_label[0] in filename_striped:
                label = angry_label[1]
            elif afraid_label[0] in filename_striped:
                label = afraid_label[1]
            elif neutral_label[0] in filename_striped:
                label = neutral_label[1]
            elif other[0] in filename_striped:
                label = other[1]
            else:
                raise Exception("Label error")

            save_folder = facial_expressions_folder + str(label) + "/"

            if not path.exists(save_folder):
                makedirs(save_folder)

            cv2.imwrite(save_folder + "/" + image_filename, face)


def create_test_set_facial_expressions():
    train_percentage = 0.7

    individuals = listdir(facial_expressions_folder + "train/")

    for individual_folder_name in individuals:
        individual_folder_path = facial_expressions_folder + "train/" + individual_folder_name + "/"
        val_individual_folder_path = facial_expressions_folder + "val/" + individual_folder_name + "/"

        if not path.exists(val_individual_folder_path):
            makedirs(val_individual_folder_path)

        image_filenames = listdir(individual_folder_path)
        num_of_images = len(image_filenames)

        # Sample a number of images to take from the folder.
        sample_images = np.random.choice(image_filenames, floor(num_of_images*train_percentage), replace=False)

        print("%s of %s" % (sample_images.__len__(), num_of_images))

        for sample_image in sample_images:
            cmd = "mv '%s' '%s'" % (individual_folder_path + sample_image, val_individual_folder_path + sample_image)
            sp.call(cmd, shell=True)

        print(individual_folder_path)


def extract_faces_from_test_image():
    """
    Method to extract faces from single image with the predicted class as the name of the file.
    :return:
    """
    image = cv2.imread("images/group3.jpg")

    faces = face_recognition.face_locations(image)

    index = 0
    for face in faces:
        top, right, bottom, left = face

        label = predict_cnn(image[top:bottom, left:right])[0]
        cv2.imwrite("test_images/%s-%s.jpg" % (label, index), image[top:bottom, left:right])

        index += 1


def make_confusion_matrix():
    """
    Method to create the confusion matrix for the CNN
    :return:
    """
    # image = cv2.imread("images/group.jpg")

    predicted = []
    actual = []
    # faces = face_recognition.face_locations(image)
    #
    # for face in faces:
    #     top, right, bottom, left = face
    #
    #     label = predict_cnn(image[top:bottom, left:right])[0]
    #     predicted.append(label)
    #
    #     print(label)
    #     show_image(image[top:bottom, left:right])
    #     tmp = input()
    #     actual.append(label if tmp == "" else tmp)

    individuals = listdir(face_folders_location + "test/")

    for individual in individuals:
        test_individual_folder_path = face_folders_location + "test/" + individual + "/"
        filenames = listdir(test_individual_folder_path)

        for filename in filenames:
            image = cv2.imread(test_individual_folder_path + filename)
            prediction = predict_cnn(image)[0]
            predicted.append(prediction)
            actual.append(individual)


    print(predicted)
    print(actual)

    return actual, predicted