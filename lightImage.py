import cv2
import numpy as np
# Load the image
img = cv2.imread(r"/Users/parthagrawal/Documents/CHECK/he1.webp")

# Convert to HSV color space
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Threshold the image to identify the yellow regions
yellow_min = (20, 20, 20)
yellow_max = (100, 255, 255)
yellow_mask = cv2.inRange(hsv, yellow_min, yellow_max)

# Convert the yellow regions to green
green_color = (255, 0, 0)
img[yellow_mask > 0] = green_color

# Save the modified image to a file
cv2.imwrite('modified_image.jpg', img)
cv2.imshow("Highlighted hypertensive retinography Regions-- Affected area is shown by color Green", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
