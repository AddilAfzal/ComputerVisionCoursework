from operator import itemgetter

from math import floor

from cnn import predict, start_training
from helper_functions import *
import cv2, subprocess as sp, numpy as np

from settings import folders_location, cnn_folders_location, face_folders_location, group_images_location, \
    group_image_faces_location, group_image_faces_location_two

recognizer = cv2.face.LBPHFaceRecognizer_create()


def show_image(img, dim):
    x, y, w, h = dim

    cv2.imshow('img', img[y:y+h, x:x+w])
    cv2.waitKey()
    cv2.destroyAllWindows()


def get_faces(img, position=False, greyscale=True, check_eyes=True):
    """
    Given an image, extract all faces.
    :param img: image, not path
    :param position: Whether to return the position or raw image.
    :return: A list of faces in grey
    """

    # Convert the image to greyscale
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the faces
    faces = face_cascade.detectMultiScale(grey, 1.1, 20)
    profile_faces = profile_face_cascade.detectMultiScale(grey, 1.1, 20)

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

    if not faces_list:
        print("No face")

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
    img = cv2.imread('group3.jpg')

    faces = get_faces(img, position=True, greyscale=False, check_eyes=False)

    for face in faces:
        x, y, w, h = face

        prediction, level = predict(img[y:y+h, x:x+w])

        # cv2.putText(img, "%s %s" % (number, (accuracy*100).round()/100), (x + 2, y + h - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), lineType=cv2.LINE_AA)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0) if level > 8 else (0, 0, 255), 3)
        cv2.putText(img, "%s - %s" % (prediction, level), (x + 4, y + h - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 170, 0), lineType=cv2.LINE_AA)

    cv2.imshow('img', cv2.resize(img, None, fx=0.5, fy=0.5))
    cv2.waitKey()
    cv2.destroyAllWindows()


def format_cnn():
    """
    For each folder, extract images from each of the videos
    :return:
    """
    individuals = listdir(folders_location)

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

                cmd = ('ffmpeg -i \'' + video_file_path + '\' -qscale:v 2 -r 10.0 \'' + cnn_image_folder_path + '_%03d-' + str(q) + '.jpg\'')
                sp.call(cmd, shell=True)

                q += 1


def create_test_set():
    train_percentage = 0.3

    individuals = listdir(face_folders_location + "train/")

    for individual_folder_name in individuals:
        individual_folder_path = face_folders_location + "train/" + individual_folder_name + "/"
        val_individual_folder_path = face_folders_location + "val/" + individual_folder_name + "/"

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
            p1, p2 = predict(face)

            cv2.imwrite('%s%s-%s.jpg' % (group_image_faces_location, p1, str(i)), face)
            i += 1


def combine_group_extract_with_cnn():
    individuals = listdir(folders_location)

    for individual in individuals:
        group_extracts = list(filter(lambda i: i[-3:] in ['jpg'], listdir(
            folders_location + individual + "/group_extract")))
        for image_name in group_extracts:
            cmd = "cp '%s' '%s'" % (folders_location + individual + "/group_extract/" + image_name, face_folders_location + "/train/" + individual + "/")
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