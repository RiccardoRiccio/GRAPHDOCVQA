# VT5_GDOC WORKING CODE WITH DYNAMIC GATING

import random
import numpy as np
import torch
import torch.nn as nn
from transformers import T5Tokenizer, T5ForConditionalGeneration
import models._model_utils as model_utils
from models._modules import CustomT5Config, SpatialEmbeddings, VisualEmbeddings
from PIL import Image
import PIL


class VT5_GDOC_CROSSATT(nn.Module):
    def __init__(self, config):
        super(VT5_GDOC_CROSSATT, self).__init__()
        print(" MODEL USE CROSS ATTENTION AND Dynamic Gating")
        # Configuration Parameters
        self.batch_size = config['batch_size']
        self.tokenizer = T5Tokenizer.from_pretrained(config['model_weights'])
        self.model = T5ForConditionalGeneration.from_pretrained(config['model_weights'])

        self.page_retrieval = config.get('page_retrieval', None)
        self.max_source_length = config.get('max_source_length', 512)

        # Custom T5 Configuration
        t5_config = CustomT5Config.from_pretrained(config['model_weights'])
        t5_config.visual_module_config = config['visual_module']

        # Embedding Modules
        self.spatial_embedding = SpatialEmbeddings(t5_config)  # Handles bounding boxes
        self.visual_embedding = VisualEmbeddings(t5_config)    # Handles visual features

        # Cross-Attention Layer for VT5 and GraphDoc
        self.cross_attention = nn.MultiheadAttention(embed_dim=t5_config.d_model, num_heads=8)

        # Dynamic Gating Network
        self.gate_net = nn.Sequential(
            nn.Linear(t5_config.d_model, 256),
            nn.ReLU(),
            nn.Linear(256, 2),
            nn.Softmax(dim=-1)  # Ensures the weights sum to 1
        )

        # Move all modules to the correct device
        device = config['device']
        self.model.to(device)
        self.spatial_embedding.to(device)
        self.visual_embedding.to(device)
        self.cross_attention.to(device)
        self.gate_net.to(device)

    def parallelize(self):
        self.model = nn.DataParallel(self.model)
        # Consider parallelizing other components if necessary
       

    def prepare_inputs_for_vqa(self, question, words, boxes, images, answers=None, graphdoc_embeds=None, graphdoc_masks=None):
        """
        Processes input into embeddings and attention masks with dynamic gating.
        """
        bs = len(words)
        prompt_text = ["question: {:s}  context: ".format(q) for q in question]
        prompt_box = [0, 0, 1000, 1000]
        eos_box = [0, 0, 0, 0]
        padding_box_value = 0

        # Process input_ids and bounding boxes
        batch_input_ids, batch_input_boxes = [], []
        longest_seq = 0

        for batch_idx in range(bs):
            tokenized_prompt = self.tokenizer(prompt_text[batch_idx])
            input_ids = tokenized_prompt.input_ids[:-1]  # Remove last token (typically <eos>)
            input_boxes = [prompt_box] * len(input_ids)

            for word, box in zip(words[batch_idx], boxes[batch_idx]):
                tokenized_word = self.tokenizer(word).input_ids[:-1]  # Tokenize word, remove <eos>
                input_ids.extend(tokenized_word)
                input_boxes.extend([box] * len(tokenized_word))  # Repeat box for each token

            # Truncate and append <eos>
            truncated_input_ids = input_ids[:self.max_source_length - 1] + [self.tokenizer.eos_token_id]
            truncated_input_boxes = np.concatenate([input_boxes[:self.max_source_length - 1], np.array([eos_box])])
            batch_input_ids.append(truncated_input_ids)
            batch_input_boxes.append(truncated_input_boxes)
            longest_seq = min(max(longest_seq, len(truncated_input_ids)), self.max_source_length)

        # Convert to tensors and pad
        tensor_input_ids = torch.full([bs, longest_seq], fill_value=self.tokenizer.pad_token_id, dtype=torch.long)
        tensor_boxes = torch.full([bs, longest_seq, 4], fill_value=padding_box_value, dtype=torch.long)
        tensor_attention_mask = torch.zeros([bs, longest_seq], dtype=torch.long)

        for batch_idx in range(bs):
            seq_len = len(batch_input_ids[batch_idx])
            tensor_input_ids[batch_idx, :seq_len] = torch.LongTensor(batch_input_ids[batch_idx])
            tensor_boxes[batch_idx, :seq_len, :] = torch.from_numpy(batch_input_boxes[batch_idx])
            tensor_attention_mask[batch_idx, :seq_len] = 1

        # Move tensors to the model's device
        tensor_input_ids = tensor_input_ids.to(self.model.device)
        tensor_boxes = tensor_boxes.to(self.model.device)
        tensor_attention_mask = tensor_attention_mask.to(self.model.device)

         # --- DEBUG: Check tensor shapes after input processing ---
        print("=== DEBUG: After input processing ===")
        print(f"tensor_input_ids.shape: {tensor_input_ids.shape}")           # Expected: [B, S_vt5]
        print(f"tensor_boxes.shape: {tensor_boxes.shape}")                 # Expected: [B, S_vt5, 4]
        print(f"tensor_attention_mask.shape: {tensor_attention_mask.shape}")  # Expected: [B, S_vt5]

        # Compute embeddings
        semantic_embedding = self.model.shared(tensor_input_ids)      # [B, S_vt5, D]
        spatial_embedding = self.spatial_embedding(tensor_boxes)      # [B, S_vt5, D]
        visual_embedding, visual_emb_mask = self.visual_embedding(images)  # [B, S_visual, D], [B, S_visual]

         # --- DEBUG: Check embedding shapes ---
        print("=== DEBUG: After computing embeddings ===")
        print(f"semantic_embedding.shape: {semantic_embedding.shape}")      # Expected: [B, S_vt5, D]
        print(f"spatial_embedding.shape: {spatial_embedding.shape}")        # Expected: [B, S_vt5, D]
        print(f"visual_embedding.shape: {visual_embedding.shape}")          # Expected: [B, S_visual, D]
        print(f"visual_emb_mask.shape: {visual_emb_mask.shape}")            # Expected: [B, S_visual]

        # Fuse semantic and spatial embeddings
        vt5_embedding = semantic_embedding + spatial_embedding         # [B, S_vt5, D]
        vt5_embedding = torch.cat([vt5_embedding, visual_embedding], dim=1)  # [B, S_vt5 + S_visual, D]
        tensor_attention_mask = torch.cat([tensor_attention_mask, visual_emb_mask], dim=1)  # [B, S_vt5 + S_visual]

         # --- DEBUG: Check fused embeddings and attention mask ---
        print("=== DEBUG: After fusing embeddings ===")
        print(f"vt5_embedding.shape: {vt5_embedding.shape}")                # Expected: [B, S_vt5 + S_visual, D]
        print(f"tensor_attention_mask.shape: {tensor_attention_mask.shape}")# Expected: [B, S_total_vt5_visual]

        if graphdoc_embeds is not None and graphdoc_masks is not None:
            graphdoc_embeds = graphdoc_embeds.to(vt5_embedding.device)
            graphdoc_masks = graphdoc_masks.to(tensor_attention_mask.device)

            # --- DEBUG: Before Cross-Attention ---
            print("=== DEBUG: Before Cross-Attention ===")
            print(f"vt5_embedding.shape: {vt5_embedding.shape}")  # Expected: [B, S_vt5 + S_visual, D]
            print(f"graphdoc_embeds.shape: {graphdoc_embeds.shape}")  # Expected: [B, S_gdoc, D]

            # Apply Cross-Attention (using transpose for PyTorch <1.8)
            vt5_attended, _ = self.cross_attention(
                vt5_embedding.transpose(0, 1),  
                graphdoc_embeds.transpose(0, 1),  
                graphdoc_embeds.transpose(0, 1)
            )
            vt5_attended = vt5_attended.transpose(0, 1)

            graphdoc_attended, _ = self.cross_attention(
                graphdoc_embeds.transpose(0, 1),  
                vt5_embedding.transpose(0, 1),  
                vt5_embedding.transpose(0, 1)
            )
            graphdoc_attended = graphdoc_attended.transpose(0, 1)

            # Compute Gating Weights
            summary_vector = vt5_attended.mean(dim=1)
            gating_weights = self.gate_net(summary_vector)

            vt5_weight = gating_weights[:, 0].unsqueeze(1).unsqueeze(2)
            graphdoc_weight = gating_weights[:, 1].unsqueeze(1).unsqueeze(2)

            # Apply Gating Weights
            vt5_embedding = vt5_weight * vt5_attended + (1 - vt5_weight) * vt5_embedding
            graphdoc_embedding = graphdoc_weight * graphdoc_attended + (1 - graphdoc_weight) * graphdoc_embeds

            # Concatenate VT5 + GraphDoc
            input_embeds = torch.cat([vt5_embedding, graphdoc_embedding], dim=1)
            tensor_attention_mask = torch.cat([tensor_attention_mask, graphdoc_masks], dim=1)
        else:
            input_embeds = vt5_embedding

        # Tokenize answers if provided
        if answers is not None:
            # Randomly select one answer from the list of possible answers for each instance
            answers = [random.choice(answer) for answer in answers]
            labels = self.tokenizer(answers, return_tensors='pt', padding=True).input_ids  # [B, L]
            labels[labels == self.tokenizer.pad_token_id] = -100  # Ignore padding tokens in loss
            labels = labels.to(self.model.device)
        else:
            labels = None

        return input_embeds, tensor_attention_mask, labels

    def forward(self, batch, return_pred_answer=False):
        # Unpack batch
        question = batch['questions']     # List of strings
        words = batch['words']            # List of lists of words
        boxes = batch['boxes']            # List of lists of bounding boxes
        images = batch['images']          # List of PIL Images or preprocessed tensors
        answers = batch['answers']        # List of lists of possible answers

        graphdoc_embeds = batch.get('graphdoc_embeds')  # [B, S_gdoc, D]
        graphdoc_masks = batch.get('graphdoc_masks')    # [B, S_gdoc]

        # Prepare inputs
        input_embeds, attention_mask, labels = self.prepare_inputs_for_vqa(
            question, words, boxes, images, answers, graphdoc_embeds, graphdoc_masks
        )

         # --- DEBUG: Check input_embeds and attention_mask before model ---
        print("=== DEBUG: Before model forward ===")
        print(f"input_embeds.shape: {input_embeds.shape}")              # Expected: [B, S_total, D]
        print(f"attention_mask.shape: {attention_mask.shape}")          # Expected: [B, S_total]


        # Forward pass through T5 model
        outputs = self.model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            labels=labels
        )

         # Handle predictions if needed
        pred_answers, pred_answers_conf = self.get_answer_from_model_output(input_embeds, attention_mask) if return_pred_answer else None

        if self.page_retrieval == 'oracle':
            pred_answer_pages = batch['answer_page_idx']

        elif self.page_retrieval == 'concat':
            pred_answer_pages = None
        else:
            pred_answer_pages = None

        return outputs, pred_answers, pred_answer_pages, pred_answers_conf

    def get_answer_from_model_output(self, input_embeds, attention_mask):
        """
        Generates answers using the model's generate method and computes confidence scores.
        """
        output = self.model.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            output_scores=True,
            return_dict_in_generate=True
        )
        pred_answers = self.tokenizer.batch_decode(output['sequences'], skip_special_tokens=True)  # List of strings
        pred_answers_conf = model_utils.get_generative_confidence(output)  # List of confidence scores

        return pred_answers, pred_answers_conf
