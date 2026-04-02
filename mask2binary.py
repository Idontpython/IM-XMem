import cv2
import os


input_folder = ''  # Replace with the actual path
output_folder = '' # output path


if not os.path.exists(output_folder):
    os.makedirs(output_folder)


for filename in os.listdir(input_folder):
    if filename.endswith('.png'):
       
        mask = cv2.imread(os.path.join(input_folder, filename))

       
        gray_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

       
        _, binary_mask = cv2.threshold(gray_mask, 1, 255, cv2.THRESH_BINARY)

       
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, binary_mask)

print("All mask images have been converted and saved！")

