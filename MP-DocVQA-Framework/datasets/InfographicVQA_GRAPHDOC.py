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

# ─── Ensure every document has at least this many OCR lines ───
MIN_OCR_LINES = 3
# ───────────────────────────────────────────────────────────────

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

def merge2d(tensors, pad_id):
    dim1 = max([s.shape[0] for s in tensors])
    dim2 = max([s.shape[1] for s in tensors])
    out = tensors[0].new(len(tensors), dim1, dim2).fill_(pad_id)
    for i, s in enumerate(tensors):
        out[i, :s.shape[0], :s.shape[1]] = s
    return out

def merge3d(tensors, pad_id):
    dim1 = max([s.shape[0] for s in tensors])
    dim2 = max([s.shape[1] for s in tensors])
    dim3 = max([s.shape[2] for s in tensors])
    out = tensors[0].new(len(tensors), dim1, dim2, dim3).fill_(pad_id)
    for i, s in enumerate(tensors):
        out[i, :s.shape[0], :s.shape[1], :s.shape[2]] = s
    return out
    
def mask1d(tensors, pad_id):
    lengths= [len(s) for s in tensors]
    out = tensors[0].new(len(tensors), max(lengths)).fill_(pad_id)
    for i, s in enumerate(tensors):
        out[i,:len(s)] = 1
    return out


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
        self.ocr_dir = Path(ocr_dir)
        self.images_dir = Path(images_dir)
        self.split = split
        self.ocr_graphdoc_dir = Path(ocr_graphdoc_dir)

        # Load QA pairs from the corresponding JSON file.
        qa_filename = f'infographicsVQA_{split}_v1.0.json'
        qa_path = self.qa_dir / qa_filename


        # Count how many times we pad the OCR lines when they are less than MIN_OCR_LINES (so each document has at least 3 lines)
        self.pad_count = 0

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
        rec = self.samples[idx]
        stem = rec['image_stem']

        # Load image.
        # Here we try to use the provided file name. If the extension is unknown, try common ones.
        # 1) Load & resize image
        img_path = self.images_dir / rec['image_local_name']
        if not img_path.exists():
            # fallback to .png
            img_path = self.images_dir / f"{stem}.png"
        img_orig = Image.open(img_path).convert('RGB')
        W, H = img_orig.size

        # resize to 512×512 for model
        img_resized = img_orig.resize((512, 512))
        img_tensor = torch.from_numpy(
            np.array(img_resized).transpose(2,0,1).astype(np.float32)
        )
        
        # Load OCR data.
        # We use the 'ocr_output_file' field to locate the corresponding OCR JSON.
        # 2) Word-level OCR
        ocr = json.load(open(self.ocr_dir / rec['ocr_output_file']))
        words = []; boxes_orig = []
        for w in ocr.get('WORD', []):
            words.append(w.get('Text', ''))
            bb = w['Geometry']['BoundingBox']
            box = [
                float(bb.get('Left',0.0)),
                float(bb.get('Top',0.0)),
                float(bb.get('Left',0.0))+float(bb.get('Width',0.0)),
                float(bb.get('Top',0.0)) +float(bb.get('Height',0.0))
            ]
            if len(box)==4:
                boxes_orig.append(box)
        if boxes_orig:
            boxes_orig = np.array(boxes_orig, dtype=np.float32)
            # scale to pixel coords w.r.t original image
            boxes_orig = np.round(
                boxes_orig * np.array([W, H, W, H], dtype=np.float32)
            ).astype(np.int64)
            # also scaled into 512×512
            scale = np.array([512/W,512/H,512/W,512/H],dtype=np.float32)
            boxes_resized = torch.from_numpy(
                np.round(boxes_orig.astype(np.float32)*scale).astype(np.int64)
            )
            boxes_orig = torch.from_numpy(boxes_orig)
        else:
            print("Len of Boxes is 0")
            boxes_orig = torch.zeros((0,4),dtype=torch.long)
            boxes_resized = torch.zeros((0,4),dtype=torch.long)



        # --- Entity-Level OCR for GraphDoc Branch ---
        # For GraphDoc, we expect a file named "{image_stem}_easyocr.json" in ocr_graphdoc_dir.
        ocr_graphdoc_filename = f"{stem}_easyocr.json"
        ocr_graphdoc_path = self.ocr_graphdoc_dir / ocr_graphdoc_filename
        with open(ocr_graphdoc_path, 'r') as f:
            ocr_graphdoc_data = json.load(f)
        # Get the "lines" key from the recognitionResults.
        lines_data = ocr_graphdoc_data["recognitionResults"][0].get("lines", [])
        lines = [L["text"] for L in lines_data]
        # print(f"[DEBUG Dataset] image='{stem}'  #OCR lines={len(lines)}")
        polys_lines = [L["boundingBox"] for L in lines_data]
        if polys_lines:
            line_boxes_orig = polys2bboxes(polys_lines)  # numpy [Nl,4]
            # ← INSERT HERE: print out the raw polygons and the unscaled boxes
            # print(f"DEBUG [Dataset __getitem__] for image '{stem}':")
            # print(f"    • Raw polygons (polys_lines) = {polys_lines}")
            # print(f"    • line_boxes_orig (pre‐scale) =\n{line_boxes_orig}")
            scale_wh = np.array([512/W,512/H,512/W,512/H],dtype=np.float32)
            line_boxes_resized = torch.from_numpy(
                np.round(line_boxes_orig.astype(np.float32)*scale_wh).astype(np.int64)
            )

            # ─────────── INSERT CLAMPING HERE ───────────
            # Clamp every coordinate into [0, 511], so there are no negatives or ≥512:
            line_boxes_resized = line_boxes_resized.clamp(min=0, max=511)
            # ────────────────────────────────────────────


            ######### CHANGES PRINT HERE #########
            # ← INSERT HERE: if any resized box coordinate is negative or ≥512, print details
            # if (line_boxes_resized < 0).any() or (line_boxes_resized >= 512).any():
            #     print(f"DEBUG [Dataset __getitem__] BAD BOX for image '{rec['image_stem']}':")
            #     print(f"    Original (pre-scale) boxes (numpy) =\n{line_boxes_orig}")
            #     print(f"    Rescaled → line_boxes_resized (before clamp) =\n{line_boxes_resized}")
            #     neg_mask = torch.where(line_boxes_resized < 0)
            #     over_mask = torch.where(line_boxes_resized >= 512)
            #     if neg_mask[0].numel() > 0:
            #         print(f"    → Negative coords at indices (row, col) = {list(zip(neg_mask[0].tolist(), neg_mask[1].tolist()))}")
            #     if over_mask[0].numel() > 0:
            #         print(f"    → ≥512 coords at indices (row, col) = {list(zip(over_mask[0].tolist(), over_mask[1].tolist()))}")
            # # ← INSERT HERE: immediately print the scaled boxes before any clamping
            # print(f"    • line_boxes_resized (just after scaling, before clamp) =\n{line_boxes_resized}")
            line_boxes_orig = torch.from_numpy(line_boxes_orig)
       
        else:
            print("Warning: No GraphDoc line-level boxes found")
            line_boxes_orig    = torch.zeros((0,4),dtype=torch.long)
            line_boxes_resized = torch.zeros((0,4),dtype=torch.long)
        
        # ─── PAD TO MIN_OCR_LINES SO EACH DOCUMENTS HAS AT LEAST 3 LINES (NODE) (BESIDE THE GLOBAL NODE) ───
        if len(lines) < MIN_OCR_LINES:
            need = MIN_OCR_LINES - len(lines)
            print(f"[Dataset PAD] image='{stem}' had only {len(lines)} lines → padding {need} dummy lines")
            self.pad_count += 1

            # pad the text list
            lines += [""] * need

            # pad the box tensors
            pad_orig = torch.zeros((need, 4), dtype=line_boxes_orig.dtype)
            pad_res  = torch.zeros((need, 4), dtype=line_boxes_resized.dtype)
            line_boxes_orig    = torch.cat([line_boxes_orig,    pad_orig], dim=0)
            line_boxes_resized = torch.cat([line_boxes_resized, pad_res ], dim=0)

# ─────────────────────────────



        # ← INSERT HERE: debug prints for image and line_boxes_rs
        # print(f"DEBUG [Dataset __getitem__]: image_resized.shape = {img_tensor.shape}, dtype = {img_tensor.dtype}")
        # if line_boxes_resized.numel() > 0:
        #     print(f"DEBUG [Dataset __getitem__]: line_boxes_rs shape = {line_boxes_resized.shape}, "
        #           f"min = {line_boxes_resized.min().item()}, max = {line_boxes_resized.max().item()}")
        # else:
        #     print("DEBUG [Dataset __getitem__]: line_boxes_rs is EMPTY")

        sample_info = {
            'question_id':           rec['question_id'],       # str, e.g. "12345"
            'question':              rec['question'],          # str, e.g. "What is the title?"
            'answers':               rec['answers'],           # List[str], e.g. ["Infographic Title", ...]

            'pil_image_orig':        img_orig,                 # PIL.Image of size (W,H), e.g. (1024×768)
            'image_resized':         img_tensor,               # torch.FloatTensor [3,512,512]
            'image_width':           W,                        # int, original width e.g. 1024
            'image_height':          H,                        # int, original height e.g. 768
            'image_name':            rec['image_stem'],        # str, e.g. "0001"

            'words':                 words,                    # List[str], len=Nw e.g. ["Infographic","Title","2025"]
            'word_boxes_original':   boxes_orig,               # torch.LongTensor [Nw,4], original coords
            'word_boxes_resized':    boxes_resized,            # torch.LongTensor [Nw,4], scaled to 512×512

            'lines':                 lines,                    # List[str], len=Nl e.g. ["Title Here","2025 Data"]
            'line_boxes':            line_boxes_orig,          # torch.LongTensor [Nl,4], original coords
            'line_boxes_rs':         line_boxes_resized,       # torch.LongTensor [Nl,4], scaled to 512×512

            
        }

        

        return sample_info

# HERE THERE IS NO PADDING (CHECK IF SHOULD BE ADDED)
# def singlepage_docvqa_collate_fn(batch):
#     batch = {k: [dic[k] for dic in batch] for k in batch[0]}  # List of dictionaries to dict of lists.

#     return batch

def singlepage_docvqa_collate_fn(batch):
    B = len(batch)
    # 1) PIL originals
    pil_images = [b['pil_image_orig'] for b in batch]
    # 2) stacked resized images
    images     = torch.stack([b['image_resized'] for b in batch], 0)

    # 3) word boxes + mask (use the same keys as in sample_info)
    w_o = [b['word_boxes_original']    for b in batch]  # <-- original name
    w_r = [b['word_boxes_resized']     for b in batch]
    word_boxes_original = merge2d(w_o, pad_id=0)
    word_boxes_resized = merge2d(w_r, pad_id=0)
    word_mask          = mask1d(w_o, pad_id=0)

    # 4) line boxes + mask
    l_o = [b['line_boxes']     for b in batch]  # sample_info uses 'line_boxes'
    l_r = [b['line_boxes_rs']  for b in batch]
    line_boxes      = merge2d(l_o, pad_id=0)
    line_boxes_rs   = merge2d(l_r, pad_id=0)
    line_mask       = mask1d(l_o, pad_id=0)

    return {
      # images
      'pil_images':    pil_images,               # list of PIL.Image
      'images':        images,                   # torch.FloatTensor [B,3,512,512]
      'image_widths':  [b['image_width']  for b in batch],
      'image_heights': [b['image_height'] for b in batch],
      'image_names':   [b['image_name']   for b in batch],

      # QA
      'question_id': [b['question_id'] for b in batch],
      'questions':    [b['question']    for b in batch],
      'answers':      [b['answers']     for b in batch],

      # word‐OCR
      'words':                   [b['words'] for b in batch],
      'word_boxes_original':     word_boxes_original,  # [B, Nw_max, 4]
      'word_boxes_resized':      word_boxes_resized,   # [B, Nw_max, 4]
      'word_mask':               word_mask,            # [B, Nw_max]

      # line‐OCR
      'lines':                   [b['lines'] for b in batch],
      'line_boxes':              line_boxes,           # [B, Nl_max, 4]
      'line_boxes_rs':           line_boxes_rs,        # [B, Nl_max, 4]
      'line_mask':               line_mask,            # [B, Nl_max]
    }
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
    # 3) Iterate and print out each field’s shape/type:
    for i in range(len(dataset)):
        sample = dataset[i]
        print(f"\n=== Sample {i} ===")
        print("question_id:        ", sample["question_id"])
        print("question:           ", sample["question"])
        print("answers:            ", sample["answers"])
        print("pil_image_orig:     ", sample["pil_image_orig"])                         # PIL.Image.Image
        print("image_resized.shape:", sample["image_resized"].shape)                   # torch.FloatTensor [3,512,512]
        print("image_width/height: ", sample["image_width"], sample["image_height"])
        print("image_name:         ", sample["image_name"])
        print("words:              ", sample["words"])
        print("word_boxes_original.shape:", sample["word_boxes_original"].shape)       # [Nw,4]
        print("word_boxes_resized.shape: ", sample["word_boxes_resized"].shape)        # [Nw,4]
        print("lines:              ", sample["lines"])
        print("line_boxes.shape:   ", sample["line_boxes"].shape)                    # [Nl,4]
        print("line_boxes_rs.shape:", sample["line_boxes_rs"].shape)                 # [Nl,4]
        # Optionally display the image:
        # sample["pil_image_orig"].show()
    
    from torch.utils.data import DataLoader
    # Quick collate‐fn test:
    loader = DataLoader(
        dataset,
        batch_size=2,
        collate_fn=singlepage_docvqa_collate_fn
    )
    batch = next(iter(loader))
    print("\n=== Collated batch keys/shapes ===")
    for k,v in batch.items():
        if torch.is_tensor(v):
            print(f"{k:20s}: {tuple(v.shape)}")
        else:
            print(f"{k:20s}: {type(v)} (len={len(v)})")

