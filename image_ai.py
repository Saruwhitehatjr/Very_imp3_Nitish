import os
import cv2
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator


def collector1(Collect_Button_pressed1):
    #collecting and saving images from webcam

    if (Collect_Button_pressed1 == "true"):
        webb_cam = cv2.VideoCapture(0)

        NumberOfFrames = 0
        captured1 = False

        while True:
            ret , frame = webb_cam.read()

            cv2.imshow("webcam" , frame)

            cv2.imwrite("./static/training_images/class1/frame" + str(NumberOfFrames) + ".jpg" , frame)

            NumberOfFrames += 1

            print(NumberOfFrames)

            if(NumberOfFrames == 100):
                break
            captured1 = "True"

        return captured1
        webb_cam.release()
        cv2.destroyAllWindows
    
    
def collector2 (Collect_Button_pressed2):
    
    if (Collect_Button_pressed2 == "true"):
        webb_cam = cv2.VideoCapture(0)

        NumberOfFrames = 0
        captured2 = False

        while True:
            ret , frame = webb_cam.read()

            cv2.imshow("webcam" , frame)

            cv2.imwrite("./static/training_images/class2/frame" + str(NumberOfFrames) + ".jpg" , frame)

            NumberOfFrames += 1

            print(NumberOfFrames)

            if(NumberOfFrames == 100):
                break
            captured2 = "True"
        
        return captured2
        webb_cam.release()
        cv2.destroyAllWindows




    #AUGMENTATION
def train(Train_button_pressed):
    Model_trained = "False"
    if Train_button_pressed == "true":

        training_data_generator = ImageDataGenerator(
            rescale = 1.0/255,
            rotation_range=40,
            width_shift_range=0.3,
            height_shift_range=0.3,
            zoom_range=0.3,
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode=('nearest')
        )
        
        training_image_directory = "./static/training_images"

        training_augmented_images = training_data_generator.flow_from_directory(
            training_image_directory,
            target_size = (180 ,180)
        )

        #preparing the model

        model = tf.keras.models.Sequential([

            # 1st Convolution & Pooling layer
            tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(180, 180, 3)),
            tf.keras.layers.MaxPooling2D(2, 2),

            # 2nd Convolution & Pooling layer
            tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2,2),

            # 3rd Convolution & Pooling layer
            tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2,2),

            # 4th Convolution & Pooling layer
            tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2,2),

            # Flatten the results to feed into a Dense Layer
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(0.5),

            # Classification Layer
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(2, activation='sigmoid')
        ])

        #training the model

        model.compile(loss = 'binary_crossentropy' , optimizer = 'adam' , metrics = ['accuracy'])
        history = model.fit(training_augmented_images , epochs=20 , validation_data = training_augmented_images , verbose = True)
        Model_trained = "True"
        
        return Model_trained

def testing():
    print("hello world")
