import warnings
warnings.filterwarnings("ignore")
import sys
import dlib
import cv2
from cv2 import *
from skimage import io
from PIL import Image
from numpy import asarray
import tensorflow as tf
import numpy as np
from io import BytesIO


MODEL = tf.keras.models.load_model('fer_16.h5', compile=False)
CLASS_NAMES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
import matplotlib.pyplot as plt
# Take the image file name from the command line
#file_name = sys.argv[1]
#file_name = 'elon_musk.jpg'
# Create a HOG face detector using the built-in dlib class
face_detector = dlib.get_frontal_face_detector()

win = dlib.image_window()

videoCaptureObject = cv2.VideoCapture(0)
result = True
while(result):
    ret,frame = videoCaptureObject.read()
    cv2.imwrite("NewPicture.jpg",frame)
    result = False

videoCaptureObject.release()
cv2.destroyAllWindows()

file_name = 'NewPicture.jpg'
image = io.imread(file_name)
# Run the HOG face detector on the image data.
# The result will be the bounding boxes of the faces in our image.
detected_faces = face_detector(image, 1)

print("I found {} face(s) in the file".format(len(detected_faces)))

# Open a window on the desktop showing the image
win.set_image(image)


# Loop through each face we found in the image
if len(detected_faces) != 0:
    for i, face_rect in enumerate(detected_faces):
        # Detected faces are returned as an object with the coordinates
        # of the top, left, right and bottom edges
        print("- Face #{} found at Left: {} Top: {} Right: {} Bottom: {}".format(i+1, face_rect.left(), face_rect.top(),
                                                                                 face_rect.right(), face_rect.bottom()))

        # Draw a box around each face we found
        win.add_overlay(face_rect)

    cropped_image = image[face_rect.top():face_rect.bottom(), face_rect.left():face_rect.right()]
    cv2.imshow("cropped", cropped_image)
    cv2.imwrite("face.jpg",cropped_image)
    cv2.waitKey(5000)
    cv2.destroyAllWindows()
    print("Shape of the cropped image (Face) : " + str(cropped_image.shape))
    # Wait until the user hits <enter> to close the window
    width = 48
    height = 48
    dim = (width, height)
    # resize image
    resized = cv2.resize(cropped_image, dim, interpolation=cv2.INTER_AREA)
    img_batch = np.expand_dims(resized, 0)
    print("Shape of resized image" + str(img_batch[0].shape))
    predictions = MODEL.predict(img_batch)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    print("The predicted class is : " + predicted_class)

else:
    pass


