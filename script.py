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

# img = cv2.imread('2.jpg')
#
#
# for (x, y, w, h) in faces:
#     cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
#     roi_gray = gray[y:y + h, x:x + w]
#     roi_color = img[y:y + h, x:x + w]
#     eyes = eye_cascade.detectMultiScale(roi_gray)
#     for (ex, ey, ew, eh) in eyes:
#         cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
#
# (x,y,w,h) = faces[0]


def main():
    img = cv2.imread('2.jpg')
    faces = get_faces(img)

    for face in faces:
        cv2.imshow('img', face)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def list_sharpness(folder="/home/addil/Desktop/computer vision/working/sorted/8/"):
    images = listdir(folder)
    images = list(filter(lambda i: i[-3:] == 'jpg', images))
    for img_name in images:
        img_path = folder + img_name
        img = cv2.imread(img_path)
        print(get_sharpness(img))


def make_folders(training_path="/home/addil/Desktop/computer vision/working/sorted/"):
    contents = listdir(training_path)

    for content in contents:
        individual_folder = training_path + content + "/"

        if path.isdir(individual_folder):
            print(individual_folder)
            individual_folder_faces = individual_folder + "faces/"
            if not path.exists(individual_folder_faces):
                makedirs(individual_folder_faces)

            faces_list = []
            for index, file_name in enumerate(list(filter(lambda i: i[-3:] == 'jpg', listdir(individual_folder)))):
                image_path = individual_folder + file_name
                img = cv2.imread(image_path)
                print(image_path)

                for face in get_faces(img):
                    faces_list.append({
                        'sharpness': get_sharpness(face),
                        'image': face,
                    })

            # save the 25 sharpest faces.
            sorted_faces = sorted(faces_list, key=itemgetter('sharpness'), reverse=True)
            q = 0
            for obj in sorted_faces[:25]:
                q += 1
                cv2.imwrite("%s%s.jpg" % (individual_folder_faces, q), obj['image'])
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
