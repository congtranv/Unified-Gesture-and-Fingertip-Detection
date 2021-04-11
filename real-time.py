import cv2
import numpy as np
from unified_detector import Fingertips
from hand_detector.detector import SOLO, YOLO

hand_detection_method = 'yolo'

if hand_detection_method is 'solo':
    hand = SOLO(weights='weights/solo.h5', threshold=0.8)
elif hand_detection_method is 'yolo':
    hand = YOLO(weights='weights/yolo.h5', threshold=0.8)
else:
    assert False, "'" + hand_detection_method + "' hand detection does not exist. use either 'solo' or 'yolo' as hand detection method"

fingertips = Fingertips(weights='weights/classes8.h5')

cam = cv2.VideoCapture(0)
print('Real-time Unified Gesture & Fingertips Detection')

while True:
    ret, image = cam.read()

    if ret is False:
        break

    # hand detection
    tl, br = hand.detect(image=image)
    list_finger = ["0", "1", "2", "3", "4", "5"]
    list_gesture = ["One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight"]

    if tl and br is not None:
        num_finger = 0;
        cropped_image = image[tl[1]:br[1], tl[0]: br[0]]
        height, width, _ = cropped_image.shape

        # gesture classification and fingertips regression
        prob, pos = fingertips.classify(image=cropped_image)
        pos = np.mean(pos, 0)

        # post-processing
        prob = np.asarray([(p >= 0.5) * 1.0 for p in prob])
        for i in range(5):
            if prob[i] >= 1.0:
                num_finger += 1

        for i in range(0, len(pos), 2):
            pos[i] = pos[i] * width + tl[0]
            pos[i + 1] = pos[i + 1] * height + tl[1]

        # drawing
        font = cv2.FONT_HERSHEY_SIMPLEX
        index = 0
        color = [(15, 15, 240), (15, 240, 155), (240, 155, 15), (240, 15, 155), (240, 15, 240)]
        image = cv2.rectangle(image, (tl[0], tl[1]), (br[0], br[1]), (0, 255, 0), 2)
        cv2.putText(image, 'num_finger: ' + list_finger[num_finger], (10, 30), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
        for c, p in enumerate(prob):
            if p > 0.5:
                image = cv2.circle(image, (int(pos[index]), int(pos[index + 1])), radius=12,
                                   color=color[c], thickness=-2)
            index = index + 2
        
        print(prob)
        ges_finger = 0
        if num_finger == 2 and prob[1] == 1 and prob[2] == 1: ges_finger = 2
        elif num_finger == 2 and prob[0] == 1 and prob[4] == 1: ges_finger = 6
        elif num_finger == 2 and prob[0] == 1 and prob[1] == 1: ges_finger = 8
        elif num_finger == 3 and prob[1] == 1 and prob[2] == 1 and prob[3] == 1: ges_finger = 3
        elif num_finger == 3 and prob[0] == 1 and prob[1] == 1 and prob[4] == 1: ges_finger = 7
        else: ges_finger = num_finger

        cv2.putText(image, 'gesture: ' + list_gesture[ges_finger-1], (10, 80), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
    
    if cv2.waitKey(1) & 0xff == 27:
        break

    # display image
    cv2.imshow('Real-time Unified Gesture & Fingertips Detection', image)

cam.release()
cv2.destroyAllWindows()
