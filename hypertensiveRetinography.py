import cv2
from google.colab.patches import cv2_imshow
# Load the image
img = cv2.imread("/content/Hypertensive_retinopathy_fundus_image-SPL_sm-2015091510234433.jpg")

# Convert to HSV color space
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Threshold the image to identify the yellow regions
yellow_min = (20, 100, 100)
yellow_max = (30, 255, 255)
yellow_mask = cv2.inRange(hsv, yellow_min, yellow_max)

# Convert the yellow regions to blue
blue_color = (255, 0, 0)
img[yellow_mask > 0] = blue_color

# Save the modified image to a file
cv2.imwrite('modified_image.jpg', img)
cv2_imshow( img)
cv2.waitKey(0)
cv2.destroyAllWindows()
