from operator import itemgetter
from os import listdir, path, makedirs
import subprocess as sp
import numpy as np

import cv2

# Cascade files loaded
face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('cascades/haarcascade_eye.xml')


def has_eyes(img):
    # Image provided should already be gray
    return eye_cascade.detectMultiScale(img).__len__() > 0


def get_faces(img):
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    faces_list = []

    for (x, y, w, h) in faces:
        face = gray[y:y + h, x:x + w]
        if has_eyes(face):
            faces_list.append(face)

    return faces_list


def get_sharpness(img):
    # Assuming image is already gray
    return cv2.Laplacian(img, cv2.CV_64F).var()


def list_sharpness(folder="/home/addil/Desktop/computer vision/working/sorted/8/"):
    images = listdir(folder)
    images = list(filter(lambda i: i[-3:] == 'jpg', images))
    for img_name in images:
        img_path = folder + img_name
        img = cv2.imread(img_path)
        print(get_sharpness(img))


def grab_additional_images(training_path="/home/addil/Desktop/computer vision/working/sorted/", num_images=1):
    """
    Function to grab an additional n number of 'sharpest' images from each video file/angle.
    :param training_path:
    :param num_images:
    :return:
    """
    contents = listdir(training_path)

    for content in contents:
        individual_folder_path = training_path + content + "/"

        if path.isdir(individual_folder_path):

            # Create the training folder if it doesn't already exist.
            individual_training_folder = individual_folder_path + "training/"
            if not path.exists(individual_training_folder):
                makedirs(individual_training_folder)

            # For each folder of pictures starting with the name 'angle', grab the n sharpest images.
            for angle_folder_name in list(filter(lambda i: i.__contains__("angle"), listdir(individual_folder_path))):
                angle_folder_path = individual_folder_path + angle_folder_name + "/"

                faces_list = []

                # Find faces in each of the images, compute sharpness and store in list.
                for index, file_name in enumerate(listdir(angle_folder_path)):
                    image_file_path = angle_folder_path + file_name
                    img = cv2.imread(image_file_path)
                    print(image_file_path)

                    for face in get_faces(img):
                        faces_list.append({
                            'sharpness': get_sharpness(face),
                            'image': face,
                        })

                # save the 25 sharpest faces.
                sorted_faces = sorted(faces_list, key=itemgetter('sharpness'), reverse=True)
                q = 0
                for obj in sorted_faces[:num_images]:
                    q += 1
                    cv2.imwrite("%s/%s-%s.jpg" % (individual_training_folder, angle_folder_name, q), obj['image'])
                    del obj['image']

                del faces_list[:]
                del faces_list


def convert_videos_to_images(training_path="/home/addil/Desktop/computer vision/working/sorted/"):
    individuals = listdir(training_path)

    for individual_folder_name in individuals:
        individual_folder_path = training_path + individual_folder_name + "/"

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

                cmd = 'ffmpeg -i \'' + video_file_path + '\' -qscale:v 2  \'' + image_folder_path + '_%03d.jpg\''
                sp.call(cmd, shell=True)

                q += 1
