import cv2
from google.colab.patches import cv2_imshow
import numpy as np

# Load the retinal image
img = cv2.imread("/content/99_right.png")

# Step 1: Preprocess the image to enhance features and remove noise
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
clahe_image = clahe.apply(gray)

# Step 2: Extract features from the image
edges = cv2.Canny(clahe_image, 100, 200)
kernel = np.ones((5,5),np.uint8)
dilation = cv2.dilate(edges,kernel,iterations = 1)

# Step 3: Display the preprocessed image and extracted features
cv2_imshow(clahe_image)
cv2_imshow(dilation)

cv2.waitKey(0)
cv2.destroyAllWindows()
