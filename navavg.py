import tensorflow as tf
import scipy.misc
from nets.pilotNet import PilotNet
import cv2
import mss
import numpy as np
import time
import random
from grabscreen import grab_screen
import keyboard
from directkeys import PressKey, ReleaseKey, W, A, S, D, E, Z, X, R, C, UP, ONE, SPACE
import math

def straight():
    PressKey(W)
    ReleaseKey(A)
    ReleaseKey(D)
    #ReleaseKey(W)
      
def left():
    if random.randrange(0,3) == 1:
        PressKey(W)
    else:
        ReleaseKey(W)
    PressKey(A)
    ReleaseKey(S)
        
def right():
    if random.randrange(0,3) == 1:
        PressKey(W)
    else:
        ReleaseKey(W)
    PressKey(D)
    ReleaseKey(A)
    ReleaseKey(S)

FLAGS = tf.app.flags.FLAGS

"""model from nvidia's training"""
tf.app.flags.DEFINE_string(
    'model', './save/model_nvidia.ckpt',
    """Path to the model parameter file.""")

tf.app.flags.DEFINE_string(
    'steer_image', './steering_wheel_image.jpg',
    """Steering wheel image to show corresponding steering wheel angle.""")

if __name__ == '__main__':
    c = []
    img = cv2.imread(FLAGS.steer_image, 0)
    rows, cols = img.shape

    with tf.Graph().as_default():
        smoothed_angle = 0
        i = 0

        # construct model
        model = PilotNet()

        saver = tf.train.Saver()
        with tf.Session() as sess:
            # restore model variables
            saver.restore(sess, FLAGS.model)

            

            while (True):
                last_time = time.time()
                screen = grab_screen(region=(340, 180, 600, 250))
                #screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
                
                image = scipy.misc.imresize(screen, [66, 200]) / 255.0

                steering = sess.run(
                    model.steering,
                    feed_dict={
                        model.image_input: [image],
                        model.keep_prob: 1.0
                    }
                )

                degrees = steering[0][0] * 180.0 / scipy.pi
                
                a = str(degrees)
                b = int(float(a))
                
                # Calculating average steering angle from 10 samples
                c.append(b)
                #print(c)
                avg = sum(c)/len(c)
                print("Average Steering Angle: {}".format(avg))
                if len(c) == 10:
                    # if list contains exceeds 10 elements the last 5 elements are retained, the rest discarded to make space for the next batch
                    c[5:9]

                # Navigation
                if avg >= -10 and avg <= 10:
                    straight()
                elif avg < -10:
                    left()
                elif avg > 10:
                    right()

                print("Predicted steering angle: " +
                        str(degrees) + " degrees")

                print("fps: {}".format(1 / (time.time() - last_time)))
                
                cv2.imshow("Neural Network Input", image)

                #cv2.putText(screen, "FPs: {}".format((1 / (time.time() - last_time)), (40+250,70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 1,cv2.LINE_AA)
                cv2.imshow("Screen Capture", screen)
                #print("Captured image size: {} x {}").format(frame.shape[0], frame.shape[1])

                # make smooth angle transitions by turning the steering wheel based on the difference of the current angle
                # and the predicted angle
                smoothed_angle += 0.2 * pow(abs((degrees - smoothed_angle)), 2.0 / 3.0) * (
                    degrees - smoothed_angle) / abs(degrees - smoothed_angle)
                M = cv2.getRotationMatrix2D(
                    (cols/2, rows/2), -smoothed_angle, 1)
                dst = cv2.warpAffine(img, M, (cols, rows))
                cv2.imshow("Steering Wheel", dst)

                i += 1

                # Press "q" to quit
                if cv2.waitKey(25) & 0xFF == ord("q"):
                    cv2.destroyAllWindows()
                    break
