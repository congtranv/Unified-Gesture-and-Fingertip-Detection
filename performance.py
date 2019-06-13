import os
import cv2
import time
import numpy as np
from statistics import mean
from net.network import model
from preprocess.label_gen_test import label_generator_testset

test_image_file = []
test_folders = ['SingleOneTest', 'SingleTwoTest', 'SingleThreeTest', 'SingleFourTest',
                'SingleFiveTest', 'SingleSixTest', 'SingleSevenTest', 'SingleEightTest']

for folder in test_folders:
    test_image_file = test_image_file + os.listdir('../EgoGesture Dataset/' + folder + '/')

print('# of images for performance analysis: ', len(test_image_file))

""" Key points Detection """
model = model()
model.summary()
model.load_weights('weights/performance.h5')


def classify(image):
    image = np.asarray(image)
    image = cv2.resize(image, (128, 128))
    image = image.astype('float32')
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    probability, position = model.predict(image)
    probability = probability[0]
    position = position[0]
    return probability, position


def class_finder(prob):
    cls = None
    # classes = ['SingleOne', 'SingleTwo', 'SingleThree', 'SingleFour', 'SingleFive',
    #            'SingleSix', 'SingleSeven', 'SingleEight']
    # index numbers
    classes = [0, 1, 2, 3, 4, 5, 6, 7]

    if np.array_equal(prob, np.array([0, 1, 0, 0, 0])):
        cls = classes[0]
    elif np.array_equal(prob, np.array([0, 1, 1, 0, 0])):
        cls = classes[1]
    elif np.array_equal(prob, np.array([0, 1, 1, 1, 0])):
        cls = classes[2]
    elif np.array_equal(prob, np.array([0, 1, 1, 1, 1])):
        cls = classes[3]
    elif np.array_equal(prob, np.array([1, 1, 1, 1, 1])):
        cls = classes[4]
    elif np.array_equal(prob, np.array([1, 0, 0, 0, 1])):
        cls = classes[5]
    elif np.array_equal(prob, np.array([1, 1, 0, 0, 1])):
        cls = classes[6]
    elif np.array_equal(prob, np.array([1, 1, 0, 0, 0])):
        cls = classes[7]

    return cls


# Classification
ground_truth_class = np.array([0, 0, 0, 0, 0, 0, 0, 0])
prediction_class = np.array([0, 0, 0, 0, 0, 0, 0, 0])

# Regression
fingertip_err = np.array([0, 0, 0, 0, 0, 0, 0, 0])
avg_time = 0
iteration = 0
conf_mat = np.zeros(shape=(8, 8))

for image_numbers, image_name in enumerate(test_image_file, 1):
    print('Images: ', image_numbers)
    image, tl, cropped_image, ground_truths = label_generator_testset(image_name=image_name, initial='Test')
    height, width, _ = cropped_image.shape
    gt_prob, gt_pos = ground_truths

    """ Predictions """
    tic = time.time()
    prob, pos = classify(image=cropped_image)
    pos = np.mean(pos, 0)

    """ Post processing """
    threshold = 0.5
    prob = np.asarray([(p >= threshold) * 1.0 for p in prob])
    for i in range(0, len(pos), 2):
        pos[i] = pos[i] * width + tl[0]
        pos[i + 1] = pos[i + 1] * height + tl[1]

    toc = time.time()
    avg_time = avg_time + (toc - tic)
    iteration = iteration + 1

    """ Calculations """
    # Classification
    gt_cls = class_finder(prob=gt_prob)
    pred_cls = class_finder(prob=prob)
    ground_truth_class[gt_cls] = ground_truth_class[gt_cls] + 1

    if gt_cls == pred_cls:
        prediction_class[pred_cls] = prediction_class[pred_cls] + 1

        # Regression
        squared_diff = np.square(gt_pos - pos)
        error = 0
        for i in range(0, 5):
            if prob[i] == 1:
                error = error + np.sqrt(squared_diff[2 * i] + squared_diff[2 * i + 1])
        error = error / sum(prob)
        fingertip_err[pred_cls] = fingertip_err[pred_cls] + error

    conf_mat[gt_cls, pred_cls] = conf_mat[gt_cls, pred_cls] + 1

    """ Drawing finger tips """
    index = 0
    color = [(15, 15, 240), (15, 240, 155), (240, 155, 15), (240, 15, 155), (240, 15, 240)]
    for c, p in enumerate(prob):
        if p == 1:
            image = cv2.circle(image, (int(pos[index]), int(pos[index + 1])), radius=12,
                               color=color[c], thickness=-2)
        index = index + 2

    # cv2.imshow('', image)
    # cv2.waitKey(0)
    # cv2.imwrite('output_perform/' + image_name, image)

accuracy = prediction_class / ground_truth_class
accuracy = accuracy * 100
accuracy = np.round(accuracy, 2)
avg_time = avg_time / iteration
fingertip_err = fingertip_err / prediction_class
fingertip_err = np.round(fingertip_err, 4)
np.save('data/conf_mat.npy', conf_mat)

print(prediction_class)
print(ground_truth_class)

print('Accuracy: ', accuracy, '%')
print('Fingertip detection error: ', fingertip_err, ' pixels')
print('Mean Error: ', mean(fingertip_err), ' pixels')
print('Total Iteration: ', iteration)
print('Mean Execution Time: ', round(avg_time, 4))
