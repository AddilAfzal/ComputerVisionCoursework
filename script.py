from operator import itemgetter
from os import listdir, path, makedirs
import subprocess as sp
from shutil import rmtree
import numpy as np
import cv2

# Cascade files loaded
face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('cascades/haarcascade_eye.xml')
profile_face_cascade = cv2.CascadeClassifier('cascades/haarcascade_profileface.xml')

folders_location = "/home/addil/Desktop/computer vision/working/sorted/"


def has_eyes(img):
    # Image provided should already be gray
    return eye_cascade.detectMultiScale(img).__len__() > 0


def get_face(img):
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the faces
    faces = face_cascade.detectMultiScale(gray, 1.2, 5)
    profie_faces = profile_face_cascade.detectMultiScale(gray, 1.2, 5)

    faces_list = []

    for (x, y, w, h) in faces:
        face = gray[y:y + h, x:x + w]
        if has_eyes(face):
            faces_list.append(face)

    if not faces_list:
        for (x, y, w, h) in profie_faces:
            face = gray[y:y + h, x:x + w]
            if has_eyes(face):
                faces_list.append(face)

    return faces_list


def get_sharpness(img):
    # Assuming image is already gray
    return cv2.Laplacian(img, cv2.CV_64F).var()


def get_faces_with_sharpness(angle_folder_path):
    faces_list = []

    # Find faces in each of the images, compute sharpness and store in list.
    j = 0
    for index, file_name in enumerate(listdir(angle_folder_path)):
        image_file_path = angle_folder_path + file_name
        img = cv2.imread(image_file_path)
        print(image_file_path)

        for face in get_face(img):
            faces_list.append({
                'sharpness': get_sharpness(face),
                'image': face,
            })
            print(j)
            j += 1

    return faces_list


def sort_and_save(faces_list, individual_training_folder, angle_folder_name, n):
    # save the n sharpest faces.
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

        if individual_folder_name:
            individual_folder_path = folders_location + individual_folder_name + "/"

            individual_training_folder = individual_folder_path + "training"
            if path.exists(individual_training_folder):
                rmtree(individual_training_folder)

            if full:
                for f in listdir(individual_folder_path):
                    if f.__contains__("angle"):
                        rmtree(individual_folder_path + f)
