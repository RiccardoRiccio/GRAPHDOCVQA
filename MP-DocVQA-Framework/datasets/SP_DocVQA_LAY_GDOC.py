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

class SPDocVQA_LAY_GDOC(Dataset):
    def __init__(self, imbd_dir, images_dir, split, kwargs, max_samples=None):
        """
        imbd_dir: Directory where the imdb .npy file is stored.
        images_dir: Directory containing the document images.
        split: Split name, e.g. 'train' or 'val'.
        kwargs: Additional settings (use_images, get_raw_ocr_data, use_graphdoc, hierarchical_method, etc.).
        max_samples: If provided, limits the dataset size.
        """

        print("USING DATSET: SP-DOCVQA__LAY_GDOC.PY")
        # Load the imdb file (NumPy format) as before.
        data = np.load(os.path.join(imbd_dir, "imdb_{:s}.npy".format(split)), allow_pickle=True)
        self.header = data[0]
        self.imdb = data[1:]
        self.hierarchical_method = kwargs.get('hierarchical_method', False)

        # Optionally limit the dataset size.
        if max_samples:
            self.imdb = self.imdb[:max_samples]

        self.max_answers = 2
        self.images_dir = images_dir

        self.use_images = kwargs.get('use_images', False)
        self.get_raw_ocr_data = kwargs.get('get_raw_ocr_data', False)

        # GraphDoc settings (unchanged from original).
        self.use_graphdoc = kwargs.get('use_graphdoc', False)
        self.graphdoc_dir = kwargs.get('graphdoc_dir', images_dir)
        missing_count = 0
        if self.use_graphdoc:
            valid_samples = []
            for record in self.imdb:
                base_name, _ = os.path.splitext(record['image_name'])
                graphdoc_filename = f"{base_name}.pt"
                graphdoc_path = os.path.join(self.graphdoc_dir, graphdoc_filename)
                if os.path.exists(graphdoc_path):
                    valid_samples.append(record)
                else:
                    missing_count += 1
                    print(f"[Warning] Missing GraphDoc embedding for {graphdoc_filename}")
            self.imdb = valid_samples
        print(f"[Info] Total missing GraphDoc embeddings for split '{split}': {missing_count}")

        # -------------------------------
        # NEW: Load the DocLayout-YOLO model once.
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
        return len(self.imdb)

    def __getitem__(self, idx):
        record = self.imdb[idx]
        question = record['question']
        context = ' '.join([word.lower() for word in record['ocr_tokens']])
        context_page_corresp = [0 for _ in range(len(context))]  # For answer page prediction (mocked as 0).

        answers = list(set(answer.lower() for answer in record['answers']))

        # Load the image if required.
        if self.use_images:
            image_name = os.path.join(self.images_dir, "{:s}.png".format(record['image_name']))
            # print(f"[DEBUG] spdocvqa_lay Processing Image Path: {image_name}") 
            image = Image.open(image_name).convert("RGB")

        # Get raw OCR data if requested.
        if self.get_raw_ocr_data:
            words = [word.lower() for word in record['ocr_tokens']]
            boxes = np.array(record['ocr_normalized_boxes'])
        if self.hierarchical_method:
            words = [words]
            boxes = [boxes]
            image_name = [image_name]
            image = [image]

        start_idxs, end_idxs = self._get_start_end_idx(context, answers)

        sample_info = {
            'question_id': record['question_id'],
            'questions': question,
            'contexts': context,
            'answers': answers,
            'start_indxs': start_idxs,
            'end_indxs': end_idxs,
            'image_names': image_name,
        }

        if self.use_images:
            sample_info['image_names'] = image_name
            sample_info['images'] = image

        if self.get_raw_ocr_data:
            sample_info['words'] = words
            sample_info['boxes'] = boxes
            sample_info['num_pages'] = 1
            sample_info['answer_page_idx'] = 0

            # -------------------------------
            # NEW: Run DocLayout-YOLO on the image to obtain detected entities.
            image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            det_res = self.doclayout_model.predict(
                image_cv,
                imgsz=1024,
                conf=0.2,
                device=device
            )
        
           

            # Before calling the predict() method, open a null file.
            # with open(os.devnull, 'w') as f:
            #     # Redirect stdout to the null file so that any prints are suppressed.
            #     with redirect_stdout(f):
            #         det_res = self.doclayout_model.predict(
            #             image_cv,
            #             imgsz=1024,
            #             conf=0.2,
            #             device=device
            #         )

            entities_boxes = []
            entities_classes = []
            if len(det_res) > 0:
                # Get the detected boxes (absolute coordinates) and class ids.
                boxes_tensor = det_res[0].boxes.xyxy.cpu().numpy()  # shape: (N, 4)
                classes_tensor = det_res[0].boxes.cls.cpu().numpy()  # shape: (N,)
                for box_val, cls in zip(boxes_tensor, classes_tensor):
                    entities_boxes.append(box_val.tolist())
                    entities_classes.append(int(cls))
            # Convert OCR normalized boxes from the dataset to absolute coordinates.
            width, height = image.size
            ocr_boxes_abs = []
            # Note: record['ocr_normalized_boxes'] is assumed to be a list of boxes for the page.
            for box in record['ocr_normalized_boxes']:
                abs_box = [box[0] * width, box[1] * height, box[2] * width, box[3] * height]
                ocr_boxes_abs.append(abs_box)
            # For each OCR token, use our new helper function to select the best entity.
            entity_tags = []
            entity_boxes = []
            for token_box in ocr_boxes_abs:
                best_entity, best_entity_box = select_entity_for_token(token_box, entities_boxes, entities_classes)
                entity_tags.append(best_entity)
                # Convert the best entity box (absolute) to normalized coordinates.
                norm_entity_box = [best_entity_box[0] / width,
                                    best_entity_box[1] / height,
                                    best_entity_box[2] / width,
                                    best_entity_box[3] / height]
                entity_boxes.append(norm_entity_box)
            # Add entity information to the sample.
            sample_info['entity_tags'] = entity_tags       # List of entity class ids (or -1 if none).
            sample_info['entity_boxes'] = entity_boxes       # List of normalized boxes [0, 1].
            # -------------------------------

        else:
            # For extractive models.
            sample_info['context_page_corresp'] = context_page_corresp
            sample_info['start_indxs'] = start_idxs
            sample_info['end_indxs'] = end_idxs

        # NEW: Load GraphDoc embeddings if enabled.
        if self.use_graphdoc:
            base_name, _ = os.path.splitext(record['image_name'])
            graphdoc_filename = f"{base_name}.pt"
            graphdoc_path = os.path.join(self.graphdoc_dir, graphdoc_filename)
            if os.path.exists(graphdoc_path):
                graphdoc_data = torch.load(graphdoc_path)
                # Assume graphdoc_data is a dictionary with 'last_hidden_state' and 'attention_mask'
                graphdoc_embeds = graphdoc_data['last_hidden_state'].squeeze(0)  # [seq_len, d_model]
                graphdoc_masks = graphdoc_data['attention_mask'].squeeze(0)      # [seq_len]
            else:
                print(f"[Warning] GraphDoc embedding not found for {graphdoc_filename}. Using placeholders.")
                graphdoc_embeds = torch.zeros(1, 768)
                graphdoc_masks = torch.zeros(1)
            sample_info['graphdoc_embeds'] = graphdoc_embeds
            sample_info['graphdoc_masks'] = graphdoc_masks
            sample_info['graphdoc_path'] = graphdoc_path
        
        # --- DEBUG PRINTS ---
        # print("=== Sample Debug Info ===")
        # print("Question:", question)
        # if self.get_raw_ocr_data:
        #     print("OCR Tokens:", words)
        #     print("OCR Normalized Boxes:", boxes)
        #     print("Computed Entity Tags:", sample_info.get('entity_tags', None))
        #     print("Computed Entity Boxes (normalized):", sample_info.get('entity_boxes', None))
        # print("=========================")
        # # --- END DEBUG PRINTS ---

        return sample_info

    def _get_start_end_idx(self, context, answers):
        answer_positions = []
        for answer in answers:
            start_idx = context.find(answer)
            if start_idx != -1:
                end_idx = start_idx + len(answer)
                answer_positions.append([start_idx, end_idx])
        if len(answer_positions) > 0:
            start_idx, end_idx = random.choice(answer_positions)
        else:
            start_idx, end_idx = 0, 0
        return start_idx, end_idx

def singlepage_docvqa_collate_fn(batch):
    batch = {k: [dic[k] for dic in batch] for k in batch[0]}  # List of dictionaries to dict of lists.

    # If graphdoc is used, stack the embeddings and masks
    if 'graphdoc_embeds' in batch and 'graphdoc_masks' in batch:
        graphdoc_embeds = batch['graphdoc_embeds']  # List of [seq_len, d_model] tensors or [B, seq_len, d_model]
        graphdoc_masks = batch['graphdoc_masks']    # List of [seq_len] tensors or [B, seq_len]

        # print("graphdoc_embeds and graphdoc_masks LOADED FROM BATCH:", graphdoc_embeds, graphdoc_masks)

        # Determine the maximum sequence length in the batch
        max_seq_len = max(embed.size(0) for embed in graphdoc_embeds)

        # Pad graphdoc_embeds and graphdoc_masks
        padded_graphdoc_embeds = pad_sequence(graphdoc_embeds, batch_first=True, padding_value=0)  # [B, max_seq_len, d_model]
        padded_graphdoc_masks = pad_sequence(graphdoc_masks, batch_first=True, padding_value=0)      # [B, max_seq_len]

        # Update the batch dictionary
        batch['graphdoc_embeds'] = padded_graphdoc_embeds
        batch['graphdoc_masks'] = padded_graphdoc_masks
        # print("AFTER COLLATE padded_graphdoc_embeds AND padded_graphdoc_masks", padded_graphdoc_embeds.shape, padded_graphdoc_masks.shape)

    return batch

# For quick testing from the command line.
if __name__ == '__main__':
    singlepage_docvqa = SPDocVQA("/SSD/Datasets/DocVQA/Task1/pythia_data/imdb/docvqa/", split='val')

