
import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import cv2
from contextlib import redirect_stdout


import json
from pathlib import Path


# -------------------------------
# Import DocLayout-YOLO model and Hugging Face Hub helper.
from doclayout_yolo import YOLOv10
from huggingface_hub import hf_hub_download
# -------------------------------

import logging
logging.getLogger("doclayout_yolo").setLevel(logging.WARNING)


def compute_iou(boxA, boxB):
    """
    Compute the Intersection over Union (IoU) between two boxes.
    Both boxes are expected in [x_min, y_min, x_max, y_max] (absolute coordinates) format.
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def select_entity_for_token(token_box, entities_boxes, entities_classes):
    """
    For a given token box, first check if it is fully contained in any of the entities.
    If multiple entities contain the token, choose the one with the smallest area.
    If none fully contain the token, fallback to selecting the entity with the highest IoU.
    Returns the best entity's class and box.
    """
    contained_entities = []
    # Check for full containment.
    for ent_box, ent_class in zip(entities_boxes, entities_classes):
        if (token_box[0] >= ent_box[0] and token_box[1] >= ent_box[1] and 
            token_box[2] <= ent_box[2] and token_box[3] <= ent_box[3]):
            contained_entities.append((ent_box, ent_class))
    
    # If one or more entities fully contain the token, choose the one with the smallest area.
    if contained_entities:
        best_ent_box, best_ent_class = min(
            contained_entities, 
            key=lambda x: (x[0][2] - x[0][0]) * (x[0][3] - x[0][1])
        )
        return best_ent_class, best_ent_box

    # Fallback: Use IoU.
    best_iou = 0.0
    best_ent_class = -1  # -1 indicates no matching entity.
    best_ent_box = [0, 0, 0, 0]
    for ent_box, ent_class in zip(entities_boxes, entities_classes):
        iou_val = compute_iou(token_box, ent_box)
        if iou_val > best_iou:
            best_iou = iou_val
            best_ent_class = ent_class
            best_ent_box = ent_box
    return best_ent_class, best_ent_box



class InfographicsVQADataset(Dataset):
    def __init__(self, imdb_dir, images_dir, ocr_dir, split, dataset_kwargs, max_samples=None):
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
        
        # -------------------------------
        # Initialize DocLayout-YOLO model by default.
        model_path = hf_hub_download(
            repo_id="juliozhao/DocLayout-YOLO-DocStructBench",
            filename="doclayout_yolo_docstructbench_imgsz1024.pt"
        )
        self.doclayout_model = YOLOv10(model_path)
        # Mapping for entity classes (as defined by DocLayout-YOLO).
        self.entity_classes = {
            0: 'title', 
            1: 'plain text', 
            2: 'abandon', 
            3: 'figure', 
            4: 'figure_caption',
            5: 'table', 
            6: 'table_caption', 
            7: 'table_footnote', 
            8: 'isolate_formula', 
            9: 'formula_caption'
        }
        # -------------------------------

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_stem = sample['image_stem']
        
        # Load OCR data.
        # We use the 'ocr_output_file' field to locate the corresponding OCR JSON.
        ocr_path = self.ocr_dir / sample['ocr_output_file']
        with open(ocr_path, 'r') as f:
            ocr_data = json.load(f)
        
        words = []
        boxes = []
        # Iterate over the "LINE" entries directly (your OCR file does not have nested "Words")
        # for line in ocr_data.get('LINE', []):
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
            if len(boxes) == 0:
                print(f"Warning: No OCR boxes found for sample {sample['question_id']}")
                boxes = np.empty((0, 4), dtype=np.float32)
            else:
                boxes = np.array(boxes, dtype=np.float32)

        # If no boxes were found, create an empty array with shape (0,4)
        if len(boxes) == 0:
            print(f"Warning: LEN OF BOXES IS '0")
            boxes = np.empty((0, 4), dtype=np.float32)
        else:
            boxes = np.array(boxes, dtype=np.float32)
        
        # Load image.
        # Here we try to use the provided file name. If the extension is unknown, try common ones.
        img_path = self.images_dir / sample['image_local_name']
        if not img_path.exists():
            # Fallback: try with .png if not found.
            img_path = self.images_dir / f"{image_stem}.png"
        image = Image.open(img_path).convert('RGB')
        width, height = image.size

        # Build the base sample dictionary.
        sample_info = {
            'question_id': sample['question_id'],
            'questions': sample['question'],
            'answers': sample['answers'],
            'words': words,
            'boxes': boxes,
            'images': image,
            'image_name': image_stem
        }

        # --- DocLayout-YOLO Entity Tagging (always enabled) ---
        # Convert OCR normalized boxes to absolute coordinates.
        ocr_boxes_abs = []
        for box in boxes:
            abs_box = [box[0] * width, box[1] * height, box[2] * width, box[3] * height]
            ocr_boxes_abs.append(abs_box)

        # Run DocLayout-YOLO on the image.
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        det_res = self.doclayout_model.predict(
            image_cv,
            imgsz=1024,
            conf=0.2,
            device=device
        )

        entities_boxes = []
        entities_classes = []
        if len(det_res) > 0:
            boxes_tensor = det_res[0].boxes.xyxy.cpu().numpy()  # shape: (N, 4)
            classes_tensor = det_res[0].boxes.cls.cpu().numpy()   # shape: (N,)
            for box_val, cls in zip(boxes_tensor, classes_tensor):
                entities_boxes.append(box_val.tolist())
                entities_classes.append(int(cls))

        # For each OCR token, select the best matching entity.
        entity_tags = []
        entity_boxes = []
        for token_box in ocr_boxes_abs:
            best_entity, best_entity_box = select_entity_for_token(token_box, entities_boxes, entities_classes)
            entity_tags.append(best_entity)
            norm_entity_box = [best_entity_box[0] / width,
                               best_entity_box[1] / height,
                               best_entity_box[2] / width,
                               best_entity_box[3] / height]
            entity_boxes.append(norm_entity_box)

        sample_info['entity_tags'] = entity_tags         # List of entity class IDs (or -1 if none)
        sample_info['entity_boxes'] = entity_boxes         # List of normalized entity boxes
        sample_info['ocr_abs_boxes'] = np.array(ocr_boxes_abs, dtype=np.float32)

        return sample_info

def singlepage_docvqa_collate_fn(batch):
    batch = {k: [dic[k] for dic in batch] for k in batch[0]}  # Convert list of dictionaries to dictionary of lists.
    return batch


if __name__ == "__main__":
    # Update these paths if necessary.
    config = {
        "imdb_dir": "/data2/users/rriccio/infographic/infographicsvqa_qas",
        "images_dir": "/data2/users/rriccio/infographic/infographicsvqa_images",
        "ocr_dir": "/data2/users/rriccio/infographic/infographicsvqa_ocr"
    }
    split = "train"
    # dataset_kwargs remains for compatibility.
    dataset_kwargs = {"ocr_dir": config["ocr_dir"]}

    dataset = InfographicsVQADataset(
        imdb_dir=config["imdb_dir"],
        images_dir=config["images_dir"],
        ocr_dir=config["ocr_dir"],
        split=split,
        dataset_kwargs=dataset_kwargs,
        max_samples=20  # Test with a small number of samples.
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
        print("Entity Tags:", sample["entity_tags"])
        print("Image Name:", sample["image_name"])
        # Optionally, display the image:
        # sample["images"].show()