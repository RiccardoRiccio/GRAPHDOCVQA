import os
import sys
import cv2
import json
import torch
import numpy as np
from torch.nn import DataParallel
sys.path.append('./')

from layoutlmft.models.graphdoc.configuration_graphdoc import GraphDocConfig
from layoutlmft.models.graphdoc.modeling_graphdoc import GraphDocForEncode
from transformers import AutoModel, AutoTokenizer

#########################################
# Helper Functions
#########################################

def read_ocr(json_path):
    """
    Read OCR data from an InfographicsVQA OCR JSON file.
    This function directly iterates over the top-level "LINE" key.
    For each line, it extracts the text and builds a rectangular polygon
    from the bounding box found under "Geometry" → "BoundingBox".
    """
    ocr_data = json.load(open(json_path, 'r'))
    polys = []      # To store bounding polygons for each line
    contents = []   # To store line texts
    for line in ocr_data.get("LINE", []):
        contents.append(line.get("Text", ""))
        bbox = line["Geometry"]["BoundingBox"]
        poly = [
            [bbox.get("Left", 0.0), bbox.get("Top", 0.0)],
            [bbox.get("Left", 0.0) + bbox.get("Width", 0.0), bbox.get("Top", 0.0)],
            [bbox.get("Left", 0.0) + bbox.get("Width", 0.0), bbox.get("Top", 0.0) + bbox.get("Height", 0.0)],
            [bbox.get("Left", 0.0), bbox.get("Top", 0.0) + bbox.get("Height", 0.0)]
        ]
        polys.append(poly)
    return polys, contents

def polys2bboxes(polys):
    """
    Convert a list of polygons (list of [x, y] pairs) into bounding boxes.
    Each bounding box is returned in [x1, y1, x2, y2] format.
    """
    bboxes = []
    for poly in polys:
        poly = np.array(poly).reshape(-1)
        x1 = poly[0::2].min()
        y1 = poly[1::2].min()
        x2 = poly[0::2].max()
        y2 = poly[1::2].max()
        bboxes.append([x1, y1, x2, y2])
    return np.array(bboxes, dtype=np.float32)

def mean_pooling(model_output, attention_mask):
    """
    Performs mean pooling on the token embeddings weighted by the attention mask.
    """
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def extract_sentence_embeddings(contents, tokenizer, sentence_bert):
    """
    Tokenizes the list of OCR texts and extracts a single embedding per line using Sentence-BERT.
    """
    encoded_input = tokenizer(contents, padding=True, truncation=True, return_tensors='pt')
    encoded_input = encoded_input.to(sentence_bert.device)
    with torch.no_grad():
        model_output = sentence_bert(**encoded_input)
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask']).cpu().numpy()
    return sentence_embeddings

def merge2d(tensors, pad_id):
    """
    Pads a list of 2D tensors to the same shape.
    """
    dim1 = max([s.shape[0] for s in tensors])
    dim2 = max([s.shape[1] for s in tensors])
    out = tensors[0].new(len(tensors), dim1, dim2).fill_(pad_id)
    for i, s in enumerate(tensors):
        out[i, :s.shape[0], :s.shape[1]] = s
    return out

def merge3d(tensors, pad_id):
    """
    Pads a list of 3D tensors (e.g., images) to the same shape.
    """
    dim1 = max([s.shape[0] for s in tensors])
    dim2 = max([s.shape[1] for s in tensors])
    dim3 = max([s.shape[2] for s in tensors])
    out = tensors[0].new(len(tensors), dim1, dim2, dim3).fill_(pad_id)
    for i, s in enumerate(tensors):
        out[i, :s.shape[0], :s.shape[1], :s.shape[2]] = s
    return out

def mask1d(tensors, pad_id):
    """
    Creates a 1D mask for a list of sequences (tensors).
    """
    lengths = [len(s) for s in tensors]
    out = tensors[0].new(len(tensors), max(lengths)).fill_(pad_id)
    for i, s in enumerate(tensors):
        out[i, :len(s)] = 1
    return out

#########################################
# Main Code: Run GraphDoc on Infographic Dataset and Save Embeddings
#########################################

def main(max_samples=5):
    # Paths for the dataset
    images_dir = "/data2/users/rriccio/infographic/infographicsvqa_images"
    ocr_dir = "/data2/users/rriccio/infographic/infographicsvqa_ocr"
    output_dir = "/data2/users/rriccio/infographic/graphdoc_embeddings_info"
    os.makedirs(output_dir, exist_ok=True)

    # Paths to pretrained models (adjust if needed)
    model_name_or_path = "/data2/users/rriccio/pretrained_model/graphdoc"
    sentence_model_path = "/data2/users/rriccio/pretrained_model/sentence-bert"

    # Initialize GraphDoc model and Sentence-BERT tokenizer/model.
    config = GraphDocConfig.from_pretrained(model_name_or_path)
    graphdoc = GraphDocForEncode.from_pretrained(model_name_or_path, config=config)
    # Wrap the model with DataParallel if multiple GPUs are available.
        # Dynamically use DataParallel if more than one GPU is available.
    if torch.cuda.device_count() > 1:
        graphdoc = DataParallel(graphdoc).cuda()
    else:
        graphdoc = graphdoc.cuda()
    graphdoc = graphdoc.eval()


    tokenizer = AutoTokenizer.from_pretrained(sentence_model_path)
    sentence_bert = AutoModel.from_pretrained(sentence_model_path)
    sentence_bert = sentence_bert.cuda().eval()

    # Define the fixed input image size for GraphDoc.
    input_H, input_W = 512, 512

    processed = 0  # Counter for processed images

    # Loop through each image file in the images directory.
    for image_file in os.listdir(images_dir):
        # Process common image file extensions.
        if not image_file.lower().endswith((".jpg", ".jpeg", ".png")):
            print("no image")
            continue

        # If max_samples is set and reached, break the loop.
        if max_samples is not None and processed >= max_samples:
            break

        image_path = os.path.join(images_dir, image_file)
        # Assume the corresponding OCR file has the same base name and a .json extension.
        ocr_file = os.path.splitext(image_file)[0] + ".json"
        ocr_path = os.path.join(ocr_dir, ocr_file)

        if not os.path.exists(ocr_path):
            print(f"OCR file not found for {image_file}. Skipping.")
            continue

        # Load and resize the image.
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load image: {image_path}. Skipping.")
            continue
        H_orig, W_orig = image.shape[:2]
        ratio_H = input_H / H_orig
        ratio_W = input_W / W_orig
        image_resized = cv2.resize(image, dsize=(input_W, input_H))

        # Read OCR data using the "LINE" key.
        polys, contents = read_ocr(ocr_path)
        if len(contents) == 0:
            print(f"No OCR lines found in {ocr_path}. Skipping.")
            continue

        # Debug: Print the extracted line tokens.
        # print(f"Image: {image_file}\nOCR Lines: {contents}")

        bboxes = polys2bboxes(polys)
        if bboxes.size == 0:
            print(f"No bounding boxes found in OCR file: {ocr_path}. Skipping.")
            continue
        # Adjust OCR bounding boxes according to the resized image.
        # bboxes[:, 0::2] = bboxes[:, 0::2] * ratio_W
        # bboxes[:, 1::2] = bboxes[:, 1::2] * ratio_H
        
        # Since OCR coordinates are normalized, multiply directly by target dimensions.
        bboxes[:, 0::2] = (bboxes[:, 0::2] * input_W).astype(np.int32)
        bboxes[:, 1::2] = (bboxes[:, 1::2] * input_H).astype(np.int32)

        # Debug: Print the adjusted bounding boxes.
        # print(f"Adjusted bounding boxes for {image_file}:\n{bboxes}")

        # Extract sentence embeddings from OCR line texts.
        sentence_embeddings = extract_sentence_embeddings(contents, tokenizer, sentence_bert)

        # Append a global node (covering the entire image) to the OCR boxes and embeddings.
        global_bbox = np.array([0, 0, input_W, input_H]).astype('int64')
        bboxes = np.concatenate([global_bbox[None, :], bboxes], axis=0)
        global_embed = np.zeros_like(sentence_embeddings[0])
        sentence_embeddings = np.concatenate([global_embed[None, :], sentence_embeddings], axis=0)

       # Build input tensors using the original merge functions.
        input_images = merge3d([torch.from_numpy(image_resized.transpose(2, 0, 1).astype(np.float32))], 0).cuda()
        input_embeds = merge2d([torch.from_numpy(sentence_embeddings)], 0).cuda()
        attention_mask = mask1d([torch.from_numpy(sentence_embeddings)], 0).cuda()
        input_bboxes = merge2d([torch.from_numpy(bboxes).long()], 0).cuda()


        input_data = dict(
            image=input_images,
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            bbox=input_bboxes,
            return_dict=True,
        )

        with torch.no_grad():
            output = graphdoc(**input_data)

        embedding_data = {
            "image_name": image_file,
            "last_hidden_state": output.last_hidden_state.cpu(),
            "pooler_output": output.pooler_output.cpu(),
            "orig_width": W_orig,
            "orig_height": H_orig,
            "attention_mask": attention_mask
            
        }

        save_path = os.path.join(output_dir, os.path.splitext(image_file)[0] + ".pt")
        torch.save(embedding_data, save_path)
        print(f"Saved embeddings for {image_file} at {save_path}")
        processed += 1

if __name__ == "__main__":
    # For debugging, set max_samples to a small number (e.g., 5). Set to None to process all images.
    main(max_samples=None)
