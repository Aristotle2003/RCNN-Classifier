# Basic Structure
# Use Selective Search to get the proposed regions
# Use AlexNet to get the features
# For every feature, we can send them to SVM to get the prediction
# Also, combine the original location with the proposed features location and send to
# Bounding box regression to output the location

# Assuming I have a directory "data_dir" with your labeled images and bounding box information

import numpy as np
import cv2
import os
import selectivesearch
from keras.applications.vgg16 import VGG16, preprocess_input
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_image(img_path):
    """Load and preprocess an image."""
    img = cv2.imread(img_path)
    img_resized = cv2.resize(img, (224, 224))  # Resize for VGG16
    img_preprocessed = preprocess_input(img_resized)
    return img_preprocessed

def selective_search(img):
    """Perform selective search on the image."""
    _, regions = selectivesearch.selective_search(img, scale=500, sigma=0.9, min_size=10)
    return regions

def calculate_deltas(proposed_box, ground_truth_box):
    """Calculate the deltas between the proposed and the ground truth boxes."""
    px, py, pw, ph = proposed_box
    gx, gy, gw, gh = ground_truth_box
    dx = (gx - px) / pw
    dy = (gy - py) / ph
    dw = np.log(gw / pw)
    dh = np.log(gh / ph)
    return [dx, dy, dw, dh]

def iou(boxA, boxB):
    """Calculate the Intersection over Union (IoU) of two bounding boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def prepare_data_for_training(data_dir, vgg_model):
    X, y, X_regression, deltas = [], [], [], []

    for class_name in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, class_name)
        for img_file in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_file)
            img = cv2.imread(img_path)
            img_lbl, regions = selectivesearch.selective_search(img, scale=500, sigma=0.9, min_size=10)

            # Assume a function `get_ground_truth_boxes` exists that returns a list of
            # (label, (x1, y1, x2, y2)) tuples for the current image
            ground_truth_boxes = get_ground_truth_boxes(img_file)

            for r in regions:
                x, y, w, h = r['rect']
                region_img = img[y:y+h, x:x+w]
                region_img_resized = cv2.resize(region_img, (224, 224))
                region_img_preprocessed = preprocess_input(np.expand_dims(region_img_resized, axis=0))
                # Use of CNN
                features = vgg_model.predict(region_img_preprocessed)

                for label, gt_box in ground_truth_boxes:
                    iou_score = iou((x, y, x+w, y+h), gt_box)
                    if iou_score > 0.5:  # Threshold to consider a match
                        X.append(features.flatten())
                        y.append(class_name)
                        delta = calculate_deltas((x, y, w, h), gt_box)
                        X_regression.append(features.flatten())
                        deltas.append(delta)
                        break  # Assume one ground truth match per region for simplicity

    return np.array(X), np.array(y), np.array(X_regression), np.array(deltas)

def train_rcnn(data_dir):
    """Train RCNN model with SVM for classification and linear regression for bounding box adjustments."""
    X, y, X_regression, deltas = prepare_data_for_training(data_dir, VGG16)

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Split data for SVM
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    # Train SVM for classification
    svm_classifier = SVC(kernel='linear', probability=True)
    svm_classifier.fit(X_train, y_train)
    print(f"SVM Training accuracy: {svm_classifier.score(X_train, y_train)}")
    print(f"SVM Test accuracy: {svm_classifier.score(X_test, y_test)}")

    # Train Linear Regression for bounding box adjustments
    regressor = LinearRegression()
    regressor.fit(X_regression, deltas)

    return svm_classifier, label_encoder, regressor


