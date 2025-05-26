'''
No padding in collate function: The comment notes this - you'll likely need padding for batch processing
Coordinate system consistency: Make sure both OCR sources use the same coordinate conventions
'''

import json
import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import os  # Only if you really need it elsewhere



def polys2bboxes(polys):
    """
    Converts a list of polygons (each as a flat list of 8 numbers)
    to a NumPy array of bounding boxes in the format [x_min, y_min, x_max, y_max].
    """
    bboxes = []
    for poly in polys:
        poly = np.array(poly).reshape(-1)
        x1 = poly[0::2].min()
        y1 = poly[1::2].min()
        x2 = poly[0::2].max()
        y2 = poly[1::2].max()
        bboxes.append([x1, y1, x2, y2])
    bboxes = np.array(bboxes).astype('int64')
    return bboxes

class InfographicsVQADataset(Dataset):
    def __init__(self, imdb_dir, images_dir, ocr_dir,  ocr_graphdoc_dir, split, dataset_kwargs, max_samples=None):
        """
        Args:
            imdb_dir (str): Path to the folder containing the QAS JSON file.
                            (Here, it is used for the QA JSON and also for OCR JSON filenames.)
            images_dir (str): Path to the folder containing image files.
            split (str): 'train' or 'val'.
            dataset_kwargs: (unused here but kept for compatibility)
            max_samples (int, optional): If provided, limits the number of samples.
        """
        # Use imdb_dir to locate the QA JSON file.
        self.qa_dir = Path(imdb_dir)
        # Assume the OCR files are stored in a separate folder if desired;
        # otherwise, you can use qa_dir as the OCR directory.
        self.ocr_dir = Path(ocr_dir)
        self.images_dir = Path(images_dir)
        self.split = split
        self.ocr_graphdoc_dir = Path(ocr_graphdoc_dir)

        # Load QA pairs from the corresponding JSON file.
        qa_filename = f'infographicsVQA_{split}_v1.0.json'
        qa_path = self.qa_dir / qa_filename

        # If the file doesn't exist and we're in validation mode, try the alternative filename.
        if not qa_path.exists() and split == "val":
            qa_filename = "infographicsVQA_val_v1.0_withQT.json"
            qa_path = self.qa_dir / qa_filename

        with open(qa_path, 'r') as f:
            data = json.load(f)
        
        self.samples = []
        for item in data['data']:
            if max_samples and len(self.samples) >= max_samples:
                break
            self.samples.append({
                'question_id': item['questionId'],
                'question': item['question'],
                'image_local_name': item['image_local_name'],  # e.g. "20471.jpeg"
                'image_stem': Path(item['image_local_name']).stem,  # e.g. "20471"
                'answers': item['answers'],
                'ocr_output_file': item['ocr_output_file']  # e.g. "20471.json"
            })


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_stem = sample['image_stem']

        # Load image.
        # Here we try to use the provided file name. If the extension is unknown, try common ones.
        img_path = self.images_dir / sample['image_local_name']
        if not img_path.exists():
            print("image path do not exist")
            # Fallback: try with .png if not found.
            img_path = self.images_dir / f"{image_stem}.png"
        image = Image.open(img_path).convert('RGB')
        width, height = image.size
        
        # Load OCR data.
        # We use the 'ocr_output_file' field to locate the corresponding OCR JSON.
        ocr_path = self.ocr_dir / sample['ocr_output_file']
        with open(ocr_path, 'r') as f:
            ocr_data = json.load(f)
        
        words = []
        boxes = []


        
        for word in ocr_data.get('WORD', []):
            # Extract the line text.
            text_line = word.get('Text', '')
            words.append(text_line)
            bbox = word['Geometry']['BoundingBox']
            # Extract the bounding box from the "Geometry" field.
            box = [
                float(bbox.get('Left', 0.0)),
                float(bbox.get('Top', 0.0)),
                float(bbox.get('Left', 0.0)) + float(bbox.get('Width', 0.0)),
                float(bbox.get('Top', 0.0)) + float(bbox.get('Height', 0.0))
            ]
            # Check if the box has exactly 4 elements.
            if len(box) == 4:
                boxes.append(box)
            else:
                print(f"Warning: Skipping box due to unexpected format: {box}")

        # If no boxes were found, create an empty array with shape (0,4)
        if len(boxes) == 0:
            print(f"Warning: LEN OF BOXES IS '0")
            boxes = np.empty((0, 4), dtype=np.float32)
        else:
            boxes = np.array(boxes, dtype=np.float32)
        # Scale boxes to pixel coordinates:
        boxes = np.round(boxes * np.array([width, height, width, height], dtype=np.float32)).astype(np.int64)




        # --- Entity-Level OCR for GraphDoc Branch ---
        # For GraphDoc, we expect a file named "{image_stem}_easyocr.json" in ocr_graphdoc_dir.
        ocr_graphdoc_filename = f"{image_stem}_easyocr.json"
        ocr_graphdoc_path = self.ocr_graphdoc_dir / ocr_graphdoc_filename
        with open(ocr_graphdoc_path, 'r') as f:
            ocr_graphdoc_data = json.load(f)
        # Get the "lines" key from the recognitionResults.
        lines_data = ocr_graphdoc_data["recognitionResults"][0].get("lines", [])
        lines = [line["text"] for line in lines_data]
        polys_lines = [line["boundingBox"] for line in lines_data]
        if len(polys_lines) > 0:
            line_boxes = polys2bboxes(polys_lines)
        else:
            line_boxes = np.empty((0, 4), dtype=np.int64)


    

        sample_info = {
            'question_id': sample['question_id'],
            'questions': sample['question'],
            'answers': sample['answers'],
            'words': words,
            'boxes': np.array(boxes, dtype=np.int64),
            'lines': lines,                    # Entity-level texts for GraphDoc branch
            'line_boxes': np.array(line_boxes, dtype=np.int64), # [num_lines, 4] (raw coordinates)
            'images': image,          # Wrap the image in a list
            'image_name': sample['image_stem'],
            'image_width': width,                     # Debug: image width
            'image_height': height                    # Debug: image height
        }

        

        return sample_info

# HERE THERE IS NO PADDING (CHECK IF SHOULD BE ADDED)
def singlepage_docvqa_collate_fn(batch):
    batch = {k: [dic[k] for dic in batch] for k in batch[0]}  # List of dictionaries to dict of lists.

    return batch


if __name__ == "__main__":
    # Update these paths if necessary
    config = {
        "imdb_dir": "/data2/users/rriccio/infographic/infographicsvqa_qas",
        "images_dir": "/data2/users/rriccio/infographic/infographicsvqa_images",
        "ocr_dir": "/data2/users/rriccio/infographic/infographicsvqa_ocr",
        "ocr_graphdoc_dir": "/data2/users/rriccio/easyocr_infographic",  # top-left, top-right, bottom right, bottom-left. entity-level OCR JSONs (with _easyocr.json naming)
    }
    split = "train"
    dataset_kwargs = {
        "ocr_dir": config["ocr_dir"],
    }
    
    dataset = InfographicsVQADataset(
        imdb_dir=config["imdb_dir"],
        images_dir=config["images_dir"],
        ocr_dir=config['ocr_dir'],
        ocr_graphdoc_dir=config["ocr_graphdoc_dir"],
        split=split,
        dataset_kwargs=dataset_kwargs,
        max_samples=3  # test with a small number of samples
    )
    
    print("Dataset length:", len(dataset))
    for i in range(len(dataset)):
        sample = dataset[i]
        print(f"\nSample {i}:")
        print("Question ID:", sample["question_id"])
        print("Question:", sample["questions"])
        print("Answers:", sample["answers"])
        print("Words:", sample["words"])
        print("Boxes:", sample["boxes"].shape)
        print("Boxes:", sample["boxes"])
        print("Lines:", sample["lines"])
        print("Line Boxes shape:", sample["line_boxes"].shape)
        print("Line Boxes shape:", sample["line_boxes"])
        print("Image Name:", sample["image_name"])
        print("Image Width:", sample["image_width"], "Height:", sample["image_height"])
        # if "graphdoc_path" in sample:
        #     print("GraphDoc Path:", sample["graphdoc_path"])
        # Optionally, show the image (if you are using an interactive session)
        # sample["images"][0].show()
