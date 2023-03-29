import tensorflow as tf
import numpy as np
from PIL import Image
from google.colab.patches import cv2_imshow 
import matplotlib.pyplot as plt

# Load the retina image
image = Image.open("/content/aao-K40.jpg")

# Preprocess the image
image = image.resize((256, 256))
image = np.array(image)
image = np.expand_dims(image, axis=0)
image = image / 255.0

# Load the pre-trained diabetic retinopathy detection model
model = tf.keras.models.load_model("/content/drive/MyDrive/Model.h5")

# Use the model to predict the areas affected by diabetic retinopathy
predictions = model.predict(image)
predicted_label = np.argmax(predictions, axis=1)[0]

output_image = Image.fromarray(np.uint8(image[0]*255))

if predicted_label == 1:
    green_color = (0, 255, 0)  # green
    mask = np.zeros((256, 256, 3), dtype=np.uint8)
    mask[predictions > 0.5] = green_color
    output_image.paste(Image.fromarray(mask), (0,0), Image.fromarray(mask))

output_image.show()
plt.imshow(np.squeeze(output_image))
cv2.waitKey(0)
cv2.destroyAllWindows()
