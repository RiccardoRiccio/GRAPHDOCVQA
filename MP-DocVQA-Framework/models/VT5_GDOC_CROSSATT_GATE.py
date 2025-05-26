

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5Tokenizer, T5ForConditionalGeneration
# Assuming these modules exist in your project.
import models._model_utils as model_utils
from models._modules import CustomT5Config, SpatialEmbeddings, VisualEmbeddings
from PIL import Image

def upsample_graphdoc_valid_tokens(graphdoc_embeds, graphdoc_mask, target_length=512):
    """
    For each sample in the batch, extract only the valid tokens from the GraphDoc embeddings,
    perform interpolation on those tokens to upsample to a fixed sequence length (target_length),
    and update the mask accordingly.

    Args:
        graphdoc_embeds (torch.Tensor): Tensor of shape [B, S, D] (e.g., [B, S, 768]).
        graphdoc_mask (torch.Tensor): Tensor of shape [B, S] with 1 for valid tokens and 0 for padding.
        target_length (int): The desired sequence length after upsampling.
        
    Returns:
        upsampled_embeds (torch.Tensor): Tensor of shape [B, target_length, D].
        upsampled_mask (torch.Tensor): Tensor of shape [B, target_length], binary mask (1 for valid).
    """
    B, S, D = graphdoc_embeds.shape
    upsampled_embeds_list = []
    upsampled_mask_list = []
    
    for i in range(B):
        # Get valid token indices for sample i
        valid_indices = (graphdoc_mask[i] == 1).nonzero(as_tuple=False).squeeze(-1)
        # print(f"\n=== DEBUG: Sample {i} ===")
        # print(f"Original graphdoc_embeds shape: {graphdoc_embeds.shape}")  # [B, S, D]
        # print(f"Original graphdoc_mask shape: {graphdoc_mask.shape}")      # [B, S]
        # print(f"Valid indices count: {valid_indices.numel()}") 
        
        if valid_indices.numel() == 0:
            print("No valid tokens found. Creating zero embeddings and mask.")
            # If no valid tokens, create zeros embeddings and a zero mask.
            upsampled_embeds_list.append(torch.zeros(target_length, D, device=graphdoc_embeds.device))
            upsampled_mask_list.append(torch.zeros(target_length, dtype=torch.long, device=graphdoc_mask.device))
            
            continue
        
        valid_embed = graphdoc_embeds[i][valid_indices]  # Shape: [n_valid, D]
        n_valid = valid_embed.shape[0]

        # print(f"Extracted valid embeddings shape: {valid_embed.shape}")  # [n_valid, D]
        # print(f"First few valid embeddings:\n {valid_embed[:5]}") 
        
        # Prepare valid_embed for interpolation:
        # F.interpolate expects input shape: [N, C, L]
        # Here, we treat the embedding dimension as channels.
        # valid_embed: [n_valid, D] --> [1, D, n_valid]
        valid_embed_t = valid_embed.transpose(0, 1).unsqueeze(0)
        
        if n_valid == 1:
            print(f"Only one valid token found. Replicating to match target length {target_length}.")
            # If only one valid token, replicate it target_length times.
            upsampled_embed_t = valid_embed_t.expand(1, D, target_length)
        else:
            # print(f"Interpolating from {n_valid} valid tokens to {target_length} target length.")
            # Interpolate along the sequence dimension
            upsampled_embed_t = F.interpolate(valid_embed_t, size=target_length, mode='linear', align_corners=False)
        
        # Convert back to shape [target_length, D]
        upsampled_embed = upsampled_embed_t.squeeze(0).transpose(0, 1)
        upsampled_embeds_list.append(upsampled_embed)

        # print(f"Upsampled embeddings shape: {upsampled_embed.shape}")  # [target_length, D]
        # print(f"First few upsampled embeddings:\n {upsampled_embed[:50]}")  # Print first 5 embeddings

        
        # For the mask, mark all positions as valid (1)
        upsampled_mask_list.append(torch.ones(target_length, dtype=torch.long, device=graphdoc_mask.device))
    
    # Stack along the batch dimension
    upsampled_embeds = torch.stack(upsampled_embeds_list, dim=0)  # Shape: [B, target_length, D]
    upsampled_mask = torch.stack(upsampled_mask_list, dim=0)        # Shape: [B, target_length]

    # print("\n=== FINAL DEBUG OUTPUT ===")
    # print(f"Final upsampled embeddings shape: {upsampled_embeds.shape}")  # [B, target_length, D]
    # print(f"Final upsampled mask shape: {upsampled_mask.shape}")  
    
    return upsampled_embeds, upsampled_mask

class VT5_GDOC_CROSSATT_GATE(nn.Module):
    def __init__(self, config):
        super(VT5_GDOC_CROSSATT_GATE, self).__init__()
        print("MODEL USES CROSS-ATTENTION GATE")
        # Configuration Parameters
        self.batch_size = config['batch_size']
        self.tokenizer = T5Tokenizer.from_pretrained(config['model_weights'])
        self.model = T5ForConditionalGeneration.from_pretrained(config['model_weights'])
        self.page_retrieval = config.get('page_retrieval', None)
        self.max_source_length = config.get('max_source_length', 512)

        # Freeze the encoder layers
        # for param in self.model.encoder.parameters():
        #     param.requires_grad = False

        # Custom T5 Configuration
        t5_config = CustomT5Config.from_pretrained(config['model_weights'])
        t5_config.visual_module_config = config['visual_module']

        # Embedding Modules
        self.spatial_embedding = SpatialEmbeddings(t5_config)  # For bounding boxes
        self.visual_embedding = VisualEmbeddings(t5_config)      # For visual features

        # Cross-Attention Layer for VT5 and GraphDoc
        self.cross_attention = nn.MultiheadAttention(embed_dim=t5_config.d_model, num_heads=8)

        # Dynamic Gating Network
        self.gate_net = nn.Sequential(
            nn.Linear(t5_config.d_model, 256),
            nn.ReLU(),
            nn.Linear(256, 2),
            nn.Softmax(dim=-1)  # Ensure the weights sum to 1
        )

        # Move modules to the specified device
        device = config['device']
        self.model.to(device)
        self.spatial_embedding.to(device)
        self.visual_embedding.to(device)
        self.cross_attention.to(device)
        self.gate_net.to(device)

    def parallelize(self):
        self.model = nn.DataParallel(self.model)
        # You may also parallelize other components if needed.

    def prepare_inputs_for_vqa(self, question, words, boxes, images, answers=None, graphdoc_embeds=None, graphdoc_masks=None):
        """
        Processes inputs into embeddings and attention masks with dynamic gating.
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
            input_ids = tokenized_prompt.input_ids[:-1]  # Remove last token (<eos>)
            input_boxes = [prompt_box] * len(input_ids)

            for word, box in zip(words[batch_idx], boxes[batch_idx]):
                tokenized_word = self.tokenizer(word).input_ids[:-1]  # Remove <eos>
                scaled_box = [int(round(coord * 1000)) for coord in box]
                input_ids.extend(tokenized_word)
                input_boxes.extend([scaled_box] * len(tokenized_word)) 

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

        # Move tensors to device
        tensor_input_ids = tensor_input_ids.to(self.model.device)
        tensor_boxes = tensor_boxes.to(self.model.device)
        tensor_attention_mask = tensor_attention_mask.to(self.model.device)

        # --- DEBUG: After input processing ---
        # print("=== DEBUG: After input processing ===")
        # print(f"tensor_input_ids.shape: {tensor_input_ids.shape}")  # [B, S_vt5]
        # print(f"tensor_boxes.shape: {tensor_boxes.shape}")          # [B, S_vt5, 4]
        # print(f"tensor_attention_mask.shape: {tensor_attention_mask.shape}")  # [B, S_vt5]

        # Compute embeddings
        semantic_embedding = self.model.shared(tensor_input_ids)  # [B, S_vt5, D]
        spatial_embedding = self.spatial_embedding(tensor_boxes)  # [B, S_vt5, D]
        visual_embedding, visual_emb_mask = self.visual_embedding(images)  # [B, S_visual, D], [B, S_visual]

        # --- DEBUG: After computing embeddings ---
        # print("=== DEBUG: After computing embeddings ===")
        # print(f"semantic_embedding.shape: {semantic_embedding.shape}")  # [B, S_vt5, D]
        # print(f"spatial_embedding.shape: {spatial_embedding.shape}")    # [B, S_vt5, D]
        # print(f"visual_embedding.shape: {visual_embedding.shape}")      # [B, S_visual, D]
        # print(f"visual_emb_mask.shape: {visual_emb_mask.shape}")        # [B, S_vt5_visual]

        # Fuse semantic and spatial embeddings, then concatenate visual embeddings.
        vt5_embedding = semantic_embedding + spatial_embedding  # [B, S_vt5, D]
        vt5_embedding = torch.cat([vt5_embedding, visual_embedding], dim=1)  # [B, S_vt5 + S_visual, D]
        tensor_attention_mask = torch.cat([tensor_attention_mask, visual_emb_mask], dim=1)  # [B, S_total]

        # --- DEBUG: After fusing embeddings ---
        # print("=== DEBUG: After fusing embeddings ===")
        # print(f"vt5_embedding.shape: {vt5_embedding.shape}")       # [B, S_total, D]
        # print(f"tensor_attention_mask.shape: {tensor_attention_mask.shape}")  # [B, S_total]

        # If GraphDoc embeddings are provided, process them.
        if graphdoc_embeds is not None and graphdoc_masks is not None:
            graphdoc_embeds = graphdoc_embeds.to(vt5_embedding.device)
            graphdoc_masks = graphdoc_masks.to(tensor_attention_mask.device)

            # Upsample GraphDoc embeddings by extracting only valid tokens and interpolating.
            graphdoc_embeds, graphdoc_masks = upsample_graphdoc_valid_tokens(graphdoc_embeds, graphdoc_masks, target_length=512)

            # --- DEBUG: Before Cross-Attention ---
            # print("=== DEBUG: Before Cross-Attention ===")
            # print(f"vt5_embedding.shape: {vt5_embedding.shape}")         # [B, S_total, D]
            # print(f"graphdoc_embeds.shape: {graphdoc_embeds.shape}")       # [B, 512, D]

            # Apply Cross-Attention: Let vt5_embedding attend to graphdoc_embeds.
            vt5_attended, _ = self.cross_attention(
                vt5_embedding.transpose(0, 1),
                graphdoc_embeds.transpose(0, 1),
                graphdoc_embeds.transpose(0, 1)
            )
            vt5_attended = vt5_attended.transpose(0, 1)

            # Also have graphdoc_embeds attend to vt5_embedding.
            graphdoc_attended, _ = self.cross_attention(
                graphdoc_embeds.transpose(0, 1),
                vt5_embedding.transpose(0, 1),
                vt5_embedding.transpose(0, 1)
            )
            graphdoc_attended = graphdoc_attended.transpose(0, 1)

            # Compute gating weights from a summary of the vt5_attended tokens.
            summary_vector = vt5_attended.mean(dim=1)
            gating_weights = self.gate_net(summary_vector)

            vt5_weight = gating_weights[:, 0].unsqueeze(1).unsqueeze(2)
            graphdoc_weight = gating_weights[:, 1].unsqueeze(1).unsqueeze(2)

            # Apply the gating weights to combine original and attended embeddings.
            vt5_embedding = vt5_weight * vt5_attended + (1 - vt5_weight) * vt5_embedding
            graphdoc_embedding = graphdoc_weight * graphdoc_attended + (1 - graphdoc_weight) * graphdoc_embeds

            # Concatenate the fused embeddings along the sequence dimension.
            input_embeds = torch.cat([vt5_embedding, graphdoc_embedding], dim=1)
            tensor_attention_mask = torch.cat([tensor_attention_mask, graphdoc_masks], dim=1)
        else:
            input_embeds = vt5_embedding

        # Tokenize answers if provided
        if answers is not None:
            # Randomly select one answer per instance
            answers = [random.choice(answer) for answer in answers]
            labels = self.tokenizer(answers, return_tensors='pt', padding=True).input_ids  # [B, L]
            labels[labels == self.tokenizer.pad_token_id] = -100  # Set padding tokens to ignore
            labels = labels.to(self.model.device)
        else:
            labels = None

        return input_embeds, tensor_attention_mask, labels

    def forward(self, batch, return_pred_answer=False):
        # Unpack the batch
        question = batch['questions']    # List[str]
        words = batch['words']           # List[List[str]]
        boxes = batch['boxes']           # List[List[bounding boxes]]
        images = batch['images']         # List[PIL Images or preprocessed tensors]
        answers = batch['answers']       # List[List[str]] possible answers

        graphdoc_embeds = batch.get('graphdoc_embeds')  # [B, S_gdoc, D]
        graphdoc_masks = batch.get('graphdoc_masks')      # [B, S_gdoc]

        # Prepare inputs
        input_embeds, attention_mask, labels = self.prepare_inputs_for_vqa(
            question, words, boxes, images, answers, graphdoc_embeds, graphdoc_masks
        )

        # --- DEBUG: Before model forward ---
        # print("=== DEBUG: Before model forward ===")
        # print(f"input_embeds.shape: {input_embeds.shape}")      # [B, S_total, D]
        # print(f"attention_mask.shape: {attention_mask.shape}")  # [B, S_total]

        # Forward pass through the T5 model
        outputs = self.model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            labels=labels
        )

        # Optionally, generate answers (if requested)
        pred_answers, pred_answers_conf = (self.get_answer_from_model_output(input_embeds, attention_mask)
                                             if return_pred_answer else (None, None))

        if self.page_retrieval == 'oracle':
            pred_answer_pages = batch['answer_page_idx']
        else:
            pred_answer_pages = None

        return outputs, pred_answers, pred_answer_pages, pred_answers_conf

    def get_answer_from_model_output(self, input_embeds, attention_mask):
        """
        Uses the model's generate method to produce answers and computes confidence scores.
        """
        output = self.model.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            output_scores=True,
            return_dict_in_generate=True
        )
        pred_answers = self.tokenizer.batch_decode(output['sequences'], skip_special_tokens=True)
        pred_answers_conf = model_utils.get_generative_confidence(output)
        return pred_answers, pred_answers_conf
