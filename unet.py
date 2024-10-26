path_before = 'before.png'
path_after = 'after.png'

import torch
import segmentation_models_pytorch as smp
import cv2
import numpy as np

# Load the pre-trained U-Net model
model = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", classes=1, activation='sigmoid')

# Load the second image (with the building)
img_with_building = cv2.imread(path_after)
img = cv2.resize(img_with_building, (256, 256))  # Resize to match the model input

# Preprocess the image
img = img.astype('float32') / 255.0
img = np.transpose(img, (2, 0, 1))  # Convert to CHW format (for PyTorch)
img = torch.tensor(img).unsqueeze(0)  # Add batch dimension

# Run inference
with torch.no_grad():
    output = model(img)
    
# Threshold the output to get the binary mask
output_np = output.squeeze().numpy()
mask = (output_np > 0.5).astype(np.uint8)

# Post-process the mask (optional)
kernel = np.ones((5, 5), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

# Overlay the mask on the original image
mask = cv2.resize(mask, (img_with_building.shape[1], img_with_building.shape[0]))  # Resize mask back to original size
highlighted = cv2.addWeighted(img_with_building, 0.8, cv2.cvtColor(mask * 255, cv2.COLOR_GRAY2BGR), 0.4, 0)

# Save the highlighted image
cv2.imwrite('detected_building.png', highlighted)

# Commenting out the display code
# cv2.imshow("Detected Building", highlighted)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

print("Image saved as 'detected_building.png'")
