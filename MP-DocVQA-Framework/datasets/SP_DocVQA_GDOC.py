
import os
import random
import torch  # Import torch for loading .pt files
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F



class SPDocVQA_GDOC(Dataset):

    def __init__(self, imbd_dir, images_dir, split, kwargs, max_samples=None):
        data = np.load(os.path.join(imbd_dir, "imdb_{:s}.npy".format(split)), allow_pickle=True)
        self.header = data[0]
        self.imdb = data[1:]
        self.hierarchical_method = kwargs.get('hierarchical_method', False)

        # Limit the dataset size
        # print(f"max sample for split: {split}:", max_samples)
        # if max_samples:
        #     self.imdb = self.imdb[:max_samples]


        self.max_answers = 2
        self.images_dir = images_dir

        self.use_images = kwargs.get('use_images', False)
        self.get_raw_ocr_data = kwargs.get('get_raw_ocr_data', False)

         # New: Initialize graphdoc-related parameters
        self.use_graphdoc = kwargs.get('use_graphdoc', False)  # Added use_graphdoc

        self.graphdoc_dir = kwargs.get('graphdoc_dir', images_dir)  # Set graphdoc_dir
        # print("kwargs:", kwargs)
        # print(f"GraphDoc directory initialized as: {self.graphdoc_dir}")
        # print("use gdoc",  self.use_graphdoc)
        # print("self.use_images", self.use_images)
        # print("self.get_raw_ocr_data ", self.get_raw_ocr_data )
        # Pre-filter samples with valid GraphDoc embeddings
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

        # Limit dataset size
        if max_samples:
            self.imdb = self.imdb[:max_samples]

        # Log the total count of missing embeddings
        print(f"[Info] Total missing GraphDoc embeddings for split '{split}': {missing_count}")


    def __len__(self):
        return len(self.imdb)

    def __getitem__(self, idx):
        record = self.imdb[idx]
        question = record['question']
        context = ' '.join([word.lower() for word in record['ocr_tokens']])
        context_page_corresp = [0 for ix in range(len(context))]  # This is used to predict the answer page in MP-DocVQA. To keep it simple, use a mock list with corresponding page to 0.

        answers = list(set(answer.lower() for answer in record['answers']))

        if self.use_images:
            image_name = os.path.join(self.images_dir, "{:s}.png".format(record['image_name']))
            image = Image.open(image_name).convert("RGB")

        if self.get_raw_ocr_data:
            words = [word.lower() for word in record['ocr_tokens']]
            boxes = np.array([bbox for bbox in record['ocr_normalized_boxes']])

        if self.hierarchical_method:
            words = [words]
            boxes = [boxes]
            image_name = [image_name]
            image = [image]

        start_idxs, end_idxs = self._get_start_end_idx(context, answers)

        sample_info = {'question_id': record['question_id'],
                       'questions': question,
                       'contexts': context,
                       'answers': answers,
                       'start_indxs': start_idxs,
                       'end_indxs': end_idxs
                       }

        if self.use_images:
            sample_info['image_names'] = image_name
            sample_info['images'] = image

        if self.get_raw_ocr_data:
            sample_info['words'] = words
            sample_info['boxes'] = boxes
            sample_info['num_pages'] = 1
            sample_info['answer_page_idx'] = 0

        else:  # Information for extractive models
            sample_info['context_page_corresp'] = context_page_corresp
            sample_info['start_indxs'] = start_idxs
            sample_info['end_indxs'] = end_idxs
        
        # if idx == 0:
        #     print("=== Debug: First sample from dataset ===")
        #     print("Question:", question)
        #     print("Context (first 200 characters):", context[:200])
        #     print("Ground Truth Answers:", record['answers'])

        # New: Load and add graphdoc embeddings if enabled
        if self.use_graphdoc:
            base_name, _ = os.path.splitext(record['image_name'])
            graphdoc_filename = f"{base_name}.pt"
            graphdoc_path = os.path.join(self.graphdoc_dir, graphdoc_filename)

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
            sample_info['graphdoc_path'] = graphdoc_path      # Add embedding path


        return sample_info

    def _get_start_end_idx(self, context, answers):

        answer_positions = []
        for answer in answers:
            start_idx = context.find(answer)

            if start_idx != -1:
                end_idx = start_idx + len(answer)
                answer_positions.append([start_idx, end_idx])

        if len(answer_positions) > 0:
            start_idx, end_idx = random.choice(answer_positions)  # If both answers are in the context. Choose one randomly.
        else:
            start_idx, end_idx = 0, 0  # If the indices are out of the sequence length they are ignored. Therefore, we set them as a very big number.

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



if __name__ == '__main__':
    singlepage_docvqa = SPDocVQA("/SSD/Datasets/DocVQA/Task1/pythia_data/imdb/docvqa/", split='val')
