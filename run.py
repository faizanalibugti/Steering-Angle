import tensorflow as tf
import mss
import numpy
import time
import scipy.misc
import model
import cv2

sess = tf.InteractiveSession()
saver = tf.train.Saver()
saver.restore(sess, "save/model.ckpt")

img = cv2.imread('steering_wheel_image.jpg',0)
rows,cols = img.shape

smoothed_angle = 0

with mss.mss() as sct:
    # Part of the screen to capture
    monitor = {"top": 0, "left": 0, "width": 640, "height": 480}

    while "Screen capturing":
        last_time = time.time()

        # Get raw pixels from the screen, save it to a Numpy array
        screen = numpy.array(sct.grab(monitor))
        screen = numpy.flip(screen[:, :, :3], 2)  
        screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)

        print("fps: {}".format(1 / (time.time() - last_time)))
        
        image = scipy.misc.imresize(screen, [66, 200]) / 255.0
        degrees = model.y.eval(feed_dict={model.x: [image], model.keep_prob: 1.0})[0][0] * 180 / scipy.pi

        print("Predicted steering angle: " + str(degrees) + " degrees")
        cv2.imshow("OpenCV/Numpy normal", screen)   
        #make smooth angle transitions by turning the steering wheel based on the difference of the current angle
        #and the predicted angle
        smoothed_angle += 0.2 * pow(abs((degrees - smoothed_angle)), 2.0 / 3.0) * (degrees - smoothed_angle) / abs(degrees - smoothed_angle)
        M = cv2.getRotationMatrix2D((cols/2,rows/2),-smoothed_angle,1)
        dst = cv2.warpAffine(img,M,(cols,rows))
        cv2.imshow("steering wheel", dst)


        # Press "q" to quit
        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break