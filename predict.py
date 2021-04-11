import cv2
import numpy as np
from hand_detector.detector import YOLO
from unified_detector import Fingertips
import matplotlib.pyplot as plt

hand = YOLO(weights='weights/yolo.h5', threshold=0.8)
fingertips = Fingertips(weights='weights/classes8.h5')

image = cv2.imread('data/sample.jpg')
#  image = cv2.imread('data/1.jpg')
#  image = cv2.imread('data/2.jpg')
#  image = cv2.imread('data/3.jpg')
#  image = cv2.imread('data/4.jpg')
#  image = cv2.imread('data/5.jpg')
#  image = cv2.imread('data/6.jpg')
#  image = cv2.imread('data/7.jpg')
#  image = cv2.imread('data/8.jpg')
tl, br = hand.detect(image=image)

plt.subplot(221)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Source")

image2 = image.copy()
image2 = cv2.rectangle(image2, (tl[0], tl[1]), (br[0], br[1]), (0, 255, 0), 2)
plt.subplot(222)
plt.imshow(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))
plt.title("YOLO_detected")
print("Position hand:",tl, br)

#plt.show()

list_finger = ["0", "1", "2", "3", "4", "5"]
list_gesture = ["One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight"]

if tl or br is not None:
    num_finger = 0
    cropped_image = image[tl[1]:br[1], tl[0]: br[0]]
    height, width, _ = cropped_image.shape

    plt.subplot(223)
    plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
    plt.title("Cropped")
    
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
            image = cv2.circle(image, (int(pos[index]), int(pos[index + 1])), radius=12, color=color[c], thickness=-2)
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
    
    # display image
    # cv2.imshow('Unified Gesture & Fingertips Detection', image)
    cv2.imwrite('data/samplew.jpg', image)
    #  cv2.imwrite('data/1w.jpg', image)
    #  cv2.imwrite('data/2w.jpg', image)
    #  cv2.imwrite('data/3w.jpg', image)
    #  cv2.imwrite('data/4w.jpg', image)
    #  cv2.imwrite('data/5w.jpg', image)
    #  cv2.imwrite('data/6w.jpg', image)
    #  cv2.imwrite('data/7w.jpg', image)
    #  cv2.imwrite('data/8w.jpg', image)
    
    plt.subplot(224)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Result')
    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25, wspace=0.35)
    plt.show()

    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()
