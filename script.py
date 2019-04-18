from operator import itemgetter
from os import listdir, path, makedirs
from helper_functions import *
import cv2, face_recognition, shutil as rmtree, subprocess as sp, numpy as np

folders_location = "/home/addil/Desktop/computer vision/working/sorted/"
recognizer = cv2.face.LBPHFaceRecognizer_create()


def show_image(img, dim):
    x, y, w, h = dim

    cv2.imshow('img', img[y:y+h, x:x+w])
    cv2.waitKey()
    cv2.destroyAllWindows()


def get_faces(img, position=False):
    """
    Given an image, extract all faces.
    :param position: Whether to return the position or raw image.
    :param img: image, not path
    :return: A list of faces in grey
    """

    # Convert the image to greyscale
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the faces
    faces = face_cascade.detectMultiScale(grey, 1.2, 5)
    profile_faces = profile_face_cascade.detectMultiScale(grey, 1.2, 5)

    # Where the faces will be stored once extracted.
    faces_list = []

    # For each frontal face detected.
    for (x, y, w, h) in faces:

        # Extract the face from the image.
        face = grey[y:y + h, x:x + w]

        eyes = number_of_eyes(face)

        # Check if this contains eyes. This is needed to prevent noise as being found as a face.
        if eyes in [1, 2]:

            # Add to the list of faces.
            faces_list.append((x, y, w, h) if position else face)
        else:
            print(eyes)

    # If there weren't any frontal faces in the image, we'll look for profile faces
    # if not faces_list:

    # For each profile face detected
    for (x, y, w, h) in profile_faces:
        face = grey[y:y + h, x:x + w]
        eyes = number_of_eyes(face)

        if eyes in [1, 2]:
            faces_list.append((x, y, w, h) if position else face)
        else:
            print(eyes)

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


def reset_folders(only=[], full=False):
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


def prepare_dataset(only=[]):
    only = list(map(str, only))

    individuals = only if only else listdir(folders_location)

    faces = []
    labels = []

    for individual_folder_name in individuals:
        individual_training_folder_path = folders_location + individual_folder_name + "/training/"

        image_paths = listdir(individual_training_folder_path)
        for image_path in image_paths:
            img = cv2.imread(individual_training_folder_path + image_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            faces.append(gray)
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
    img = cv2.imread('group2.jpg')

    faces = get_faces(img, position=True)

    for face in faces:
        x, y, w, h = face
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)

        id, accuracy = recognise_face(img=img[y:y+h, x:x+w])

        cv2.putText(img, "%s %s" % (id, accuracy), (x, y + h), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)

    cv2.imshow('img', cv2.resize(img, None, fx=0.5, fy=0.5))
    cv2.waitKey()
    cv2.destroyAllWindows()
