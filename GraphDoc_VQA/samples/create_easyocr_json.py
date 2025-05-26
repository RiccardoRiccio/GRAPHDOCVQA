########################################################
# R: This code is to create the easyocr json file from the image
########################################################

import easyocr
import cv2
import json
from pathlib import Path
import os

def easyocr_to_json(image_path, word_results, paragraph_results):
    # Read image to get dimensions
    img = cv2.imread(image_path)
    height, width = img.shape[:2]

    json_data = {
        "status": "Succeeded",
        "recognitionResults": [
            {
                "page": 1,
                "clockwiseOrientation": 0,
                "width": width,
                "height": height,
                "unit": "pixel",
                "lines": [],  # for paragraphs/regions
                "words": []   # for individual words
            }
        ]
    }

    # Add paragraph/region level results
    for result in paragraph_results:
        bbox, text = result
        flat_bbox = [int(coord) for point in bbox for coord in point]
        
        line = {
            "boundingBox": flat_bbox,
            "text": text
        }
        json_data["recognitionResults"][0]["lines"].append(line)

    # Add word level results
    for result in word_results:
        bbox, text, conf = result
        flat_bbox = [int(coord) for point in bbox for coord in point]
        
        word = {
            "boundingBox": flat_bbox,
            "text": text,
            "confidence": float(conf)
        }
        json_data["recognitionResults"][0]["words"].append(word)

    return json_data

def process_images(input_dir, output_dir):
    # Initialize the OCR reader
    reader = easyocr.Reader(['en'])
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Loop through all image files in the input directory
    for image_file in os.listdir(input_dir):
        image_path = os.path.join(input_dir, image_file)

        # Ensure it's a valid image file
        if not image_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            continue  # Skip non-image files
        
        file_name = Path(image_path).stem  # Extract filename without extension
        output_json_path = os.path.join(output_dir, f"{file_name}_easyocr.json")

        # Check if JSON already exists, skip if it does
        if os.path.exists(output_json_path):
            print(f"🔹 Skipping {image_file}, already processed.")
            continue

        print(f"Processing: {image_path}")

        try:
            # Get word-level results with smaller width_ths
            word_results = reader.readtext(
                image_path,
                paragraph=False,
                width_ths=0.1  # Smaller value to prevent word merging
            )
            
            # Get paragraph-level results with default parameters
            paragraph_results = reader.readtext(
                image_path,
                paragraph=True
            )

            # Convert to JSON format
            json_data = easyocr_to_json(image_path, word_results, paragraph_results)

            # Define output JSON path inside the new directory
            # file_name = Path(image_path).stem  # Extract filename without extension
            # output_json_path = os.path.join(output_dir, f"{file_name}_easyocr.json")

            # Save JSON to file
            with open(output_json_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, ensure_ascii=False, indent=2)

            print(f"✔ OCR results saved to: {output_json_path}")

        except Exception as e:
            print(f"❌ Error processing {image_path}: {str(e)}")

def main():
    input_dir = "/data2/users/rriccio/infographic/infographicsvqa_images"
    output_dir = "/data2/users/rriccio/easyocr_infographic"

    process_images(input_dir, output_dir)

if __name__ == "__main__":
    main()
#
# def main():
#     # Initialize the OCR reader
#     reader = easyocr.Reader(['en'])

#     # Create the directory if it does not exist
#     os.makedirs(output_dir, exist_ok=True)
    
#     # Path to your image
#     image_path = "/data2/users/rriccio/spdocvqa_images/ffbf0023_4.png"
    
#     # Get word-level results with smaller width_ths
#     word_results = reader.readtext(
#         image_path,
#         paragraph=False,
#         width_ths=0.1  # Smaller value to prevent word merging
#     )
    
#     # Get paragraph-level results with default parameters
#     paragraph_results = reader.readtext(
#         image_path,
#         paragraph=True,
#         # width_ths=0.7,
#         # add_margin=0.1
#     )
    
#     # Convert to JSON format
#     json_data = easyocr_to_json(image_path, word_results, paragraph_results)
    
#     # Define output directory
#     output_dir = "/data2/users/rriccio/infographic/easyocr_info"

    

#     # Define output JSON path inside the new directory
#     file_name = Path(image_path).stem  # Extract filename without extension
#     output_json_path = os.path.join(output_dir, f"{file_name}_easyocr.json")

#     # Save JSON to file
#     with open(output_json_path, 'w', encoding='utf-8') as f:
#         json.dump(json_data, f, ensure_ascii=False, indent=2)
    
#     print(f"OCR results saved to: {output_json_path}")

# if __name__ == "__main__":
#     main()