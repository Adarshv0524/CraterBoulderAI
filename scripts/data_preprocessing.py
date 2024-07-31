import cv2
import os
import numpy as np

def advanced_noise_reduction(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for filename in os.listdir(input_folder):
        if filename.endswith((".png", ".jpg", ".jpeg")):
            image_path = os.path.join(input_folder, filename)
            image = cv2.imread(image_path)
            
            if image is None:
                print(f"Error reading image: {filename}")
                continue
            
            # Apply Non-Local Means Denoising
            denoised_color = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
            
            # Convert to float32 for further processing
            denoised_float = np.float32(denoised_color) / 255.0
            
            # Apply Bilateral Filter for edge-preserving smoothing
            bilateral = cv2.bilateralFilter(denoised_float, 9, 75, 75)
            
            # Apply Median Blur to remove salt-and-pepper noise
            median = cv2.medianBlur(np.uint8(bilateral * 255), 5)
            
            # Convert back to float32
            final_denoised = np.float32(median) / 255.0
            
            # Enhance image details using unsharp masking
            gaussian = cv2.GaussianBlur(final_denoised, (0, 0), 2.0)
            unsharp_mask = cv2.addWeighted(final_denoised, 1.5, gaussian, -0.5, 0)
            
            # Clip values to ensure they are in the valid range [0, 1]
            final_result = np.clip(unsharp_mask, 0, 1)
            
            # Convert back to uint8 for saving
            final_result_uint8 = np.uint8(final_result * 255)
            
            # Save the denoised image
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, final_result_uint8)
            
            print(f"Processed: {filename}")

def contrast_enhancement(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for filename in os.listdir(input_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(input_folder, filename)
            image = cv2.imread(image_path)
            # Convert to YUV color space
            yuv_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
            # Enhance the Y channel (luminance)
            yuv_image[:,:,0] = cv2.equalizeHist(yuv_image[:,:,0])
            # Convert back to BGR color space
            enhanced_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR)
            # Save the enhanced image
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, enhanced_image)



# data_preprocessing.py

def image_segmentation(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for filename in os.listdir(input_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(input_folder, filename)
            image = cv2.imread(image_path)
            # Convert to grayscale
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Apply thresholding to segment features
            _, segmented_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
            # Save the segmented image
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, segmented_image)




if __name__ == "__main__":
    advanced_noise_reduction('../data/raw', '../data/processed/denoised')
    contrast_enhancement('../data/processed/denoised', '../data/processed/contrast_enhanced')
    image_segmentation('../data/processed/contrast_enhanced', '../data/processed/segmented')
