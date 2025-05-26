# import json
# import torch
# from pathlib import Path
# from PIL import Image
# from torch.utils.data import Dataset
# import numpy as np

# class InfographicsVQADataset(Dataset):
#     def __init__(self, imdb_dir, images_dir, ocr_dir, split, dataset_kwargs, max_samples=None):
#         """
#         Args:
#             imdb_dir (str): Path to the folder containing the QAS JSON file.
#                             (Here, it is used for the QA JSON and also for OCR JSON filenames.)
#             images_dir (str): Path to the folder containing image files.
#             split (str): 'train' or 'val'.
#             dataset_kwargs: (unused here but kept for compatibility)
#             max_samples (int, optional): If provided, limits the number of samples.
#         """
#         # Use imdb_dir to locate the QA JSON file.
#         self.qa_dir = Path(imdb_dir)
#         # Assume the OCR files are stored in a separate folder if desired;
#         # otherwise, you can use qa_dir as the OCR directory.
#         self.ocr_dir = Path(ocr_dir)
#         self.images_dir = Path(images_dir)
#         self.split = split
        
#         # Load QA pairs from the corresponding JSON file.
#         qa_filename = f'infographicsVQA_{split}_v1.0.json'
#         qa_path = self.qa_dir / qa_filename

#         # If the file doesn't exist and we're in validation mode, try the alternative filename.
#         if not qa_path.exists() and split == "val":
#             qa_filename = "infographicsVQA_val_v1.0_withQT.json"
#             qa_path = self.qa_dir / qa_filename

#         with open(qa_path, 'r') as f:
#             data = json.load(f)
        
#         self.samples = []
#         for item in data['data']:
#             if max_samples and len(self.samples) >= max_samples:
#                 break
#             self.samples.append({
#                 'question_id': item['questionId'],
#                 'question': item['question'],
#                 'image_local_name': item['image_local_name'],  # e.g. "20471.jpeg"
#                 'image_stem': Path(item['image_local_name']).stem,  # e.g. "20471"
#                 'answers': item['answers'],
#                 'ocr_output_file': item['ocr_output_file']  # e.g. "20471.json"
#             })

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
#         sample = self.samples[idx]
#         image_stem = sample['image_stem']
        
#         # Load OCR data.
#         # We use the 'ocr_output_file' field to locate the corresponding OCR JSON.
#         ocr_path = self.ocr_dir / sample['ocr_output_file']
#         with open(ocr_path, 'r') as f:
#             ocr_data = json.load(f)
        
#         words = []
#         boxes = []
#         # Iterate over the "LINE" entries directly (your OCR file does not have nested "Words")
#         # for line in ocr_data.get('LINE', []):
#         for word in ocr_data.get('WORD', []):
#             # Extract the line text.
#             text_line = word.get('Text', '')
#             words.append(text_line)
#             bbox = word['Geometry']['BoundingBox']
#             # Extract the bounding box from the "Geometry" field.
#             box = [
#                 float(bbox.get('Left', 0.0)),
#                 float(bbox.get('Top', 0.0)),
#                 float(bbox.get('Left', 0.0)) + float(bbox.get('Width', 0.0)),
#                 float(bbox.get('Top', 0.0)) + float(bbox.get('Height', 0.0))
#             ]
#             # Check if the box has exactly 4 elements.
#             if len(box) == 4:
#                 boxes.append(box)
#             else:
#                 print(f"Warning: Skipping box due to unexpected format: {box}")

#         # If no boxes were found, create an empty array with shape (0,4)
#         if len(boxes) == 0:
#             print(f"Warning: LEN OF BOXES IS '0")
#             boxes = np.empty((0, 4), dtype=np.float32)
#         else:
#             boxes = np.array(boxes, dtype=np.float32)
        
#         # Load image.
#         # Here we try to use the provided file name. If the extension is unknown, try common ones.
#         img_path = self.images_dir / sample['image_local_name']
#         if not img_path.exists():
#             # Fallback: try with .png if not found.
#             img_path = self.images_dir / f"{image_stem}.png"
#         image = Image.open(img_path).convert('RGB')

#         return {
#             'question_id': sample['question_id'],
#             'questions': sample['question'],
#             'answers': sample['answers'],
#             'words': words,
#             'boxes': np.array(boxes, dtype=np.float32),
#             'images': image,          # Wrap the image in a list
#             'image_name': sample['image_stem']
#         }
# def singlepage_docvqa_collate_fn(batch):
#     batch = {k: [dic[k] for dic in batch] for k in batch[0]}  # Convert list of dictionaries to dictionary of lists.
#     return batch


# if __name__ == "__main__":
#     # Update these paths if necessary
#     config = {
#         "imdb_dir": "/data2/users/rriccio/infographic/infographicsvqa_qas",
#         "images_dir": "/data2/users/rriccio/infographic/infographicsvqa_images",
#         "ocr_dir": "/data2/users/rriccio/infographic/infographicsvqa_ocr"
#     }
#     split = "train"
#     dataset_kwargs = {"ocr_dir": config["ocr_dir"]}
    
#     dataset = InfographicsVQADataset(
#         imdb_dir=config["imdb_dir"],
#         images_dir=config["images_dir"],
#         ocr_dir=config['ocr_dir'],
#         split=split,
#         dataset_kwargs=dataset_kwargs,
#         max_samples=20  # test with a small number of samples
#     )
    
#     print("Dataset length:", len(dataset))
#     for i in range(len(dataset)):
#         sample = dataset[i]
#         print(f"\nSample {i}:")
#         print("Question ID:", sample["question_id"])
#         print("Question:", sample["questions"])
#         print("Answers:", sample["answers"])
#         print("Words:", sample["words"])
#         print("Boxes:", sample["boxes"].shape)
#         print("Image Name:", sample["image_name"])
#         # Optionally, show the image (if you are using an interactive session)
#         # sample["images"][0].show()





import json
import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import os  # Only if you really need it elsewhere


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

         # New: GraphDoc parameters
        self.use_graphdoc = dataset_kwargs.get('use_graphdoc', False)
        self.graphdoc_dir = Path(dataset_kwargs.get('graphdoc_dir', imdb_dir))  # default to imdb_dir if not provided

        
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


        missing_count = 0
        if self.use_graphdoc:
            valid_samples = []
            for record in self.samples:
                base_name = record['image_stem']
                graphdoc_filename = f"{base_name}.pt"
                graphdoc_path = self.graphdoc_dir / graphdoc_filename

                if os.path.exists(graphdoc_path):
                    valid_samples.append(record)
                else:
                    missing_count += 1
                    print(f"[Warning] Missing GraphDoc embedding for {graphdoc_filename}")
            self.samples = valid_samples
            print(f"[Info] Total missing GraphDoc embeddings: {missing_count}")

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
        
        # Load image.
        # Here we try to use the provided file name. If the extension is unknown, try common ones.
        img_path = self.images_dir / sample['image_local_name']
        if not img_path.exists():
            # Fallback: try with .png if not found.
            img_path = self.images_dir / f"{image_stem}.png"
        image = Image.open(img_path).convert('RGB')

        sample_info = {
            'question_id': sample['question_id'],
            'questions': sample['question'],
            'answers': sample['answers'],
            'words': words,
            'boxes': np.array(boxes, dtype=np.float32),
            'images': image,          # Wrap the image in a list
            'image_name': sample['image_stem']
        }

        if self.use_graphdoc:
            graphdoc_filename = f"{sample['image_stem']}.pt"
            graphdoc_path = self.graphdoc_dir / graphdoc_filename

            # print(f"[DEBUG] GraphDoc path for {sample['image_stem']}: {graphdoc_path}")

            

            # Debug print for paths
            # print(f"Sample Index: {idx}")
            # print(f"Image path: {image_name}")
            # print(f"GraphDoc embedding path: {graphdoc_path}")

            # # Debug print
            # print(f"Constructed GraphDoc path: {graphdoc_path}")

            if os.path.exists(graphdoc_path):
                graphdoc_data = torch.load(graphdoc_path)  # Load the .pt file
                # Assuming graphdoc_data is a dictionary with 'last_hidden_state' and 'attention_mask'
                # print ("graphdoc_embeds and graphdoc_masks before squeezing, expected: [B, seq_len, d_model]",graphdoc_data['last_hidden_state'].shape, graphdoc_data['attention_mask'].shape)
                graphdoc_embeds = graphdoc_data['last_hidden_state'].squeeze(0)  # [seq_len, d_model]
                graphdoc_masks = graphdoc_data['attention_mask'].squeeze(0)      # [seq_len]
                # print ("graphdoc_embeds and graphdoc_masks AFTER squeezing, expected: [seq_len, d_model]", graphdoc_embeds.shape, graphdoc_masks.shape)
            else:
                # raise FileNotFoundError(f"GraphDoc embedding not found for {graphdoc_filename}")
                print(f"[Warning] GraphDoc embedding not found for {graphdoc_filename}. Using default placeholders.")
                graphdoc_embeds = torch.zeros(1, 768)  # Adjust dimensions to match your model
                graphdoc_masks = torch.zeros(1)

            sample_info['graphdoc_embeds'] = graphdoc_embeds  # [seq_len, d_model]
            sample_info['graphdoc_masks'] = graphdoc_masks    # [seq_len]
            sample_info['graphdoc_path'] = str(graphdoc_path)      # Add embedding path

        return sample_info


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


if __name__ == "__main__":
    # Update these paths if necessary
    config = {
        "imdb_dir": "/data2/users/rriccio/infographic/infographicsvqa_qas",
        "images_dir": "/data2/users/rriccio/infographic/infographicsvqa_images",
        "ocr_dir": "/data2/users/rriccio/infographic/infographicsvqa_ocr"
    }
    split = "train"
    dataset_kwargs = {
        "ocr_dir": config["ocr_dir"],
        "use_graphdoc": True,
        "graphdoc_dir": "/data2/users/rriccio/infographic/graphdoc_embeddings_info"
    }
    
    dataset = InfographicsVQADataset(
        imdb_dir=config["imdb_dir"],
        images_dir=config["images_dir"],
        ocr_dir=config['ocr_dir'],
        split=split,
        dataset_kwargs=dataset_kwargs,
        max_samples=25  # test with a small number of samples
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
        print("Image Name:", sample["image_name"])
        if "graphdoc_path" in sample:
            print("GraphDoc Path:", sample["graphdoc_path"])
        # Optionally, show the image (if you are using an interactive session)
        # sample["images"][0].show()
