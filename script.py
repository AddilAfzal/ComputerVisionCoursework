from os import listdir, path, makedirs

import numpy as np
import cv2

# Cascade files loaded
face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('cascades/haarcascade_eye.xml')


def get_faces(img):
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the faces
    faces = face_cascade.detectMultiScale(gray)

    faces_list = []

    for (x, y, w, h) in faces:
        faces_list.append(gray[y:y + h, x:x + w])

    return faces_list


def is_blurry(img):
    #https://stackoverflow.com/questions/39685757/how-to-make-a-new-filter-and-apply-it-on-an-image-using-cv2-in-python2-7

    # Create a dummy input image.
    canvas = np.zeros((100, 100), dtype=np.uint8)
    canvas = cv2.circle(canvas, (50, 50), 20, (255,), -1)

    kernel = np.array([[-1, -1, -1],
                       [-1, 4, -1],
                       [-1, -1, -1]])

    dst = cv2.filter2D(canvas, -1, kernel)


def get_sharpness(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

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

            q = 0
            for index, file_name in enumerate(list(filter(lambda i: i[-3:] == 'jpg', listdir(individual_folder)))):
                image_path = individual_folder + file_name
                img = cv2.imread(image_path)
                print(image_path)
                for face in get_faces(img):
                    q += 1
                    cv2.imwrite("%s%s.jpg" % (individual_folder_faces, q), face)
