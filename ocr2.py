import cv2
import numpy as np

def get_better_ocr_system(image_path):
    try:
        # Load the image
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Error: Image not found at {image_path}")

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Remove noise
        denoised = cv2.fastNlMeansDenoising(gray, h=30)

        # Apply threshold
        _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Invert if needed (ensure black text on white background)
        if np.mean(thresh[:50, :50]) < 127:
            thresh = cv2.bitwise_not(thresh)

        # Perform OCR
        extracted_text = pytesseract.image_to_string(thresh, config='--oem 3 --psm 6')

        return extracted_text.strip()

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# Test the function
image_file = '/content/hamid-roshaan-IGVGEFQHczg-unsplash.jpg'
extracted_text = get_better_ocr_system(image_file)

if extracted_text:
    print(extracted_text)