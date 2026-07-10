import cv2
import os
import numpy as np
import hashlib

#  Dataset Path
# Update this path before running the script
dataset_path = "path_to_custom_dataset"
# Threshold Values
BLUR_THRESHOLD = 80
DARK_THRESHOLD = 30
BRIGHT_THRESHOLD = 230
# Blur Detection Function
def is_blurry(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() < BLUR_THRESHOLD
# Lighting Check Function
def is_bad_lighting(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean = np.mean(gray)
    return mean < DARK_THRESHOLD or mean > BRIGHT_THRESHOLD
# Duplicate Detection (Hashing)
def get_hash(path):
    with open(path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

total_removed = 0
# Main Cleaning Loop
for class_name in os.listdir(dataset_path):

    class_path = os.path.join(dataset_path, class_name)

    if not os.path.isdir(class_path):
        continue

    print(f"\n Cleaning class: {class_name}")

    hashes = set()

    for img_name in os.listdir(class_path):

        img_path = os.path.join(class_path, img_name)

        try:
            img = cv2.imread(img_path)

            if img is None:
                os.remove(img_path)
                total_removed += 1
                continue

            if is_blurry(img):
                os.remove(img_path)
                total_removed += 1
                continue

            if is_bad_lighting(img):
                os.remove(img_path)
                total_removed += 1
                continue

            img_hash = get_hash(img_path)
            if img_hash in hashes:
                os.remove(img_path)
                total_removed += 1
                continue
            else:
                hashes.add(img_hash)

            img = cv2.resize(img, (64,64))
            cv2.imwrite(img_path, img)

        except:
            os.remove(img_path)
            total_removed += 1

print(f"\n Cleaning Complete! Removed {total_removed} bad images")
