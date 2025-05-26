import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5Tokenizer, T5ForConditionalGeneration
import models._model_utils as model_utils
from models._modules import CustomT5Config, SpatialEmbeddings, VisualEmbeddings
from PIL import Image

#############################################
# 1. Learnable Upsampler Module Definition  #
#############################################

class LearnableUpsampler(nn.Module):
    def __init__(self, input_dim, target_length=512, num_heads=4):
        super(LearnableUpsampler, self).__init__()
        self.target_length = target_length
        self.input_dim = input_dim
        # Learnable queries: shape [target_length, input_dim]
        self.queries = nn.Parameter(torch.randn(target_length, input_dim))
        # Remove batch_first; older PyTorch doesn't support it.
        self.attn = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads)

    def forward(self, valid_embeds, valid_mask):
        # valid_embeds: [B, n_valid, D]
        B, n_valid, D = valid_embeds.shape
        print(f"[LearnableUpsampler] Input valid_embeds shape: {valid_embeds.shape}")
        # Expand queries for each sample: [B, target_length, D]
        queries = self.queries.unsqueeze(0).expand(B, -1, -1)
        # print(f"[LearnableUpsampler] Expanded queries shape: {queries.shape}")
        # Transpose to [L, B, D] as required by older MultiheadAttention
        queries_t = queries.transpose(0, 1)           # [target_length, B, D]
        valid_embeds_t = valid_embeds.transpose(0, 1)   # [n_valid, B, D]
        # Create key_padding_mask: shape [B, n_valid] (True indicates a padded token)
        key_padding_mask = (valid_mask == 0)
        # Apply attention: output shape [target_length, B, D]
        upsampled_t, _ = self.attn(query=queries_t, key=valid_embeds_t, value=valid_embeds_t,
                                   key_padding_mask=key_padding_mask)
        # Transpose back to [B, target_length, D]
        upsampled_embeds = upsampled_t.transpose(0, 1)
        # print(f"[LearnableUpsampler] Output upsampled_embeds shape: {upsampled_embeds.shape}")
        return upsampled_embeds


#############################################
# 2. The Main Model with Fusion & Upsampling  #
#############################################

class VT5_GDOC_MLPFUSION(nn.Module):
    def __init__(self, config):
        super(VT5_GDOC_MLPFUSION, self).__init__()
        print("MODEL USES CROSS-ATTENTION, LEARNABLE UPSAMPLER, & EXTRA FUSION")
        # Configuration Parameters
        self.batch_size = config['batch_size']
        self.tokenizer = T5Tokenizer.from_pretrained(config['model_weights'])
        self.model = T5ForConditionalGeneration.from_pretrained(config['model_weights'])
        self.page_retrieval = config.get('page_retrieval', None)
        self.max_source_length = config.get('max_source_length', 512)
        
        # Custom T5 Configuration
        t5_config = CustomT5Config.from_pretrained(config['model_weights'])
        t5_config.visual_module_config = config['visual_module']
        
        # Embedding Modules (for text, layout, and image features)
        self.spatial_embedding = SpatialEmbeddings(t5_config)
        self.visual_embedding = VisualEmbeddings(t5_config)
        
        # Cross-Attention Layer for VT5 and GraphDoc
        self.cross_attention = nn.MultiheadAttention(embed_dim=t5_config.d_model, num_heads=8)

        
        # Learnable Upsampler for GraphDoc:
        # We assume GraphDoc embeddings are provided as [B, S_gdoc, D] (with S_gdoc variable),
        # and we want to upsample to a fixed length (e.g., 512 tokens).
        self.graphdoc_upsampler = LearnableUpsampler(input_dim=t5_config.d_model, target_length=512, num_heads=4)
        
        # Extra Fusion Block: after concatenation of T5 and upsampled GraphDoc embeddings,
        # we pass the joint representation through a small MLP.
        self.fusion_projection = nn.Sequential(
            nn.Linear(t5_config.d_model, t5_config.d_model),
            nn.ReLU(),
            nn.Linear(t5_config.d_model, t5_config.d_model)
        )
        
        # Move modules to device
        device = config['device']
        self.model.to(device)
        self.spatial_embedding.to(device)
        self.visual_embedding.to(device)
        self.cross_attention.to(device)
        self.graphdoc_upsampler.to(device)
        self.fusion_projection.to(device)
    
    def parallelize(self):
        self.model = nn.DataParallel(self.model)
        # You may also parallelize other components if needed.
    
    def prepare_inputs_for_vqa(self, question, words, boxes, images, answers=None,
                               graphdoc_embeds=None, graphdoc_masks=None):
        """
        Processes inputs into embeddings and attention masks.
        """
        bs = len(words)
        prompt_text = ["question: {:s}  context: ".format(q) for q in question]
        prompt_box = [0, 0, 1000, 1000]
        eos_box = [0, 0, 0, 0]
        padding_box_value = 0

        # Process input_ids and bounding boxes for the text input
        batch_input_ids, batch_input_boxes = [], []
        longest_seq = 0

        for batch_idx in range(bs):
            tokenized_prompt = self.tokenizer(prompt_text[batch_idx])
            input_ids = tokenized_prompt.input_ids[:-1]
            input_boxes = [prompt_box] * len(input_ids)
            for word, box in zip(words[batch_idx], boxes[batch_idx]):
                tokenized_word = self.tokenizer(word).input_ids[:-1]
                input_ids.extend(tokenized_word)
                input_boxes.extend([box] * len(tokenized_word))
            truncated_input_ids = input_ids[:self.max_source_length - 1] + [self.tokenizer.eos_token_id]
            truncated_input_boxes = np.concatenate([input_boxes[:self.max_source_length - 1], np.array([eos_box])])
            batch_input_ids.append(truncated_input_ids)
            batch_input_boxes.append(truncated_input_boxes)
            longest_seq = min(max(longest_seq, len(truncated_input_ids)), self.max_source_length)
        
        tensor_input_ids = torch.full([bs, longest_seq], fill_value=self.tokenizer.pad_token_id, dtype=torch.long)
        tensor_boxes = torch.full([bs, longest_seq, 4], fill_value=padding_box_value, dtype=torch.long)
        tensor_attention_mask = torch.zeros([bs, longest_seq], dtype=torch.long)
        for batch_idx in range(bs):
            seq_len = len(batch_input_ids[batch_idx])
            tensor_input_ids[batch_idx, :seq_len] = torch.LongTensor(batch_input_ids[batch_idx])
            tensor_boxes[batch_idx, :seq_len, :] = torch.from_numpy(batch_input_boxes[batch_idx])
            tensor_attention_mask[batch_idx, :seq_len] = 1
        
        tensor_input_ids = tensor_input_ids.to(self.model.device)
        tensor_boxes = tensor_boxes.to(self.model.device)
        tensor_attention_mask = tensor_attention_mask.to(self.model.device)
        
        # print(f"[VT5] tensor_input_ids.shape: {tensor_input_ids.shape}")
        # print(f"[VT5] tensor_boxes.shape: {tensor_boxes.shape}")
        # print(f"[VT5] tensor_attention_mask.shape: {tensor_attention_mask.shape}")
        # Compute embeddings for the text and visual modalities.
        semantic_embedding = self.model.shared(tensor_input_ids)         # [B, S_text, D]
        spatial_embedding = self.spatial_embedding(tensor_boxes)         # [B, S_text, D]
        visual_embedding, visual_emb_mask = self.visual_embedding(images)  # [B, S_visual, D], [B, S_visual]


        # print(f"[VT5] semantic_embedding.shape: {semantic_embedding.shape}")
        # print(f"[VT5] spatial_embedding.shape: {spatial_embedding.shape}")
        # print(f"[VT5] visual_embedding.shape: {visual_embedding.shape}")
        # print(f"[VT5] visual_emb_mask.shape: {visual_emb_mask.shape}")
        
        # Fuse semantic and spatial embeddings, then concatenate visual embeddings.
        vt5_embedding = semantic_embedding + spatial_embedding            # [B, S_text, D]
        vt5_embedding = torch.cat([vt5_embedding, visual_embedding], dim=1)   # [B, S_text+S_visual, D]
        tensor_attention_mask = torch.cat([tensor_attention_mask, visual_emb_mask], dim=1)  # [B, S_total]

        # print(f"[VT5] vt5_embedding.shape (after concat): {vt5_embedding.shape}")
        # print(f"[VT5] Updated attention mask shape: {tensor_attention_mask.shape}")
        
        # If GraphDoc embeddings are provided, upsample and fuse them.
        if graphdoc_embeds is not None and graphdoc_masks is not None:
            graphdoc_embeds = graphdoc_embeds.to(vt5_embedding.device)
            graphdoc_masks = graphdoc_masks.to(tensor_attention_mask.device)
            # print(f"[GraphDoc] Original graphdoc_embeds.shape: {graphdoc_embeds.shape}")  # e.g. [B, 289, D]
            # print(f"[GraphDoc] Original graphdoc_masks.shape: {graphdoc_masks.shape}")    # e.g. [B, 289]
            
            
            # --- Learnable Upsampling ---
            # First, extract valid tokens from GraphDoc.
            # For each sample, we assume graphdoc_masks has 1 for valid tokens.
            # Here, we simply use all tokens where the mask is 1.
            valid_graphdoc_embeds = []
            valid_graphdoc_masks = []
            for i in range(graphdoc_embeds.shape[0]):
                valid_indices = (graphdoc_masks[i] == 1).nonzero(as_tuple=False).squeeze(-1)
                print(f"[GraphDoc] Sample {i} valid token count: {valid_indices.numel()}")
                if valid_indices.numel() == 0:
                    # If no valid token exists, use a zero tensor.
                    valid_graphdoc_embeds.append(torch.zeros(1, graphdoc_embeds.size(-1), device=graphdoc_embeds.device))
                    valid_graphdoc_masks.append(torch.ones(1, device=graphdoc_embeds.device, dtype=torch.long))
                else:
                    valid_graphdoc_embeds.append(graphdoc_embeds[i][valid_indices])
                    valid_graphdoc_masks.append(torch.ones(valid_indices.numel(), device=graphdoc_embeds.device, dtype=torch.long))
            # Pad the valid tokens (if necessary) to allow batching.
            # Here we simply pack them into a list and then use the learnable upsampler, which
            # can handle variable lengths because we pass the valid mask.
            # First, pad each sample to the maximum n_valid across the batch.
            max_valid = max(x.shape[0] for x in valid_graphdoc_embeds)
            padded_valid = []
            padded_mask = []
            for embeds, mask in zip(valid_graphdoc_embeds, valid_graphdoc_masks):
                n_valid = embeds.shape[0]
                if n_valid < max_valid:
                    pad = torch.zeros(max_valid - n_valid, embeds.size(-1), device=embeds.device)
                    embeds = torch.cat([embeds, pad], dim=0)
                    mask = torch.cat([mask, torch.zeros(max_valid - n_valid, device=mask.device, dtype=torch.long)], dim=0)
                padded_valid.append(embeds.unsqueeze(0))
                padded_mask.append(mask.unsqueeze(0))
            valid_graphdoc_embeds = torch.cat(padded_valid, dim=0)  # [B, max_valid, D]
            valid_graphdoc_masks = torch.cat(padded_mask, dim=0)      # [B, max_valid]
            # print(f"[GraphDoc] Padded valid_graphdoc_embeds.shape: {valid_graphdoc_embeds.shape}")
            # print(f"[GraphDoc] Padded valid_graphdoc_masks.shape: {valid_graphdoc_masks.shape}")
            
            
            # Apply the learnable upsampler to obtain a fixed-length sequence.
            upsampled_graphdoc = self.graphdoc_upsampler(valid_graphdoc_embeds, valid_graphdoc_masks)  # [B, 512, D]
            # print(f"[GraphDoc] Upsampled graphdoc.shape: {upsampled_graphdoc.shape}")  # Expected [B, 512, D]
            # Create a mask for the upsampled tokens (all valid in the output).
            upsampled_graphdoc_mask = torch.ones(upsampled_graphdoc.shape[:2], device=graphdoc_masks.device, dtype=torch.long)
            
            # Now, you can fuse (for example, via cross-attention) or simply concatenate.
            # Here we concatenate the VT5 (text+visual) embeddings with the upsampled GraphDoc embeddings.
            joint_embeds = torch.cat([vt5_embedding, upsampled_graphdoc], dim=1)
            joint_attention_mask = torch.cat([tensor_attention_mask, upsampled_graphdoc_mask], dim=1)
            # print(f"[Fusion] joint_embeds.shape (after concat): {joint_embeds.shape}")
            # print(f"[Fusion] joint_attention_mask.shape: {joint_attention_mask.shape}")
        else:
            joint_embeds = vt5_embedding
            joint_attention_mask = tensor_attention_mask
        
        # After concatenation, apply an extra learnable fusion/projection layer.
        fused_embeds = self.fusion_projection(joint_embeds)  # [B, (S_total + 512), D]
        print(f"[Fusion] fused_embeds.shape (after projection): {fused_embeds.shape}")
        
        # If answers are provided, tokenize and produce labels.
        if answers is not None:
            answers = [random.choice(answer) for answer in answers]
            labels = self.tokenizer(answers, return_tensors='pt', padding=True).input_ids
            labels[labels == self.tokenizer.pad_token_id] = -100
            labels = labels.to(self.model.device)
        else:
            labels = None
        
        return fused_embeds, joint_attention_mask, labels
    
    def forward(self, batch, return_pred_answer=False):
        question = batch['questions']
        words = batch['words']
        boxes = batch['boxes']
        images = batch['images']
        answers = batch['answers']
        graphdoc_embeds = batch.get('graphdoc_embeds')
        graphdoc_masks = batch.get('graphdoc_masks')
        
        input_embeds, attention_mask, labels = self.prepare_inputs_for_vqa(
            question, words, boxes, images, answers, graphdoc_embeds, graphdoc_masks
        )

        print(f"[Model] Final input_embeds.shape: {input_embeds.shape}")
        print(f"[Model] Final attention_mask.shape: {attention_mask.shape}")
        
        outputs = self.model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            labels=labels
        )
        
        pred_answers, pred_answers_conf = (self.get_answer_from_model_output(input_embeds, attention_mask)
                                             if return_pred_answer else (None, None))
        
        if self.page_retrieval == 'oracle':
            pred_answer_pages = batch['answer_page_idx']
        else:
            pred_answer_pages = None
        
        return outputs, pred_answers, pred_answer_pages, pred_answers_conf
    
    def get_answer_from_model_output(self, input_embeds, attention_mask):
        output = self.model.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            output_scores=True,
            return_dict_in_generate=True
        )
        pred_answers = self.tokenizer.batch_decode(output['sequences'], skip_special_tokens=True)
        pred_answers_conf = model_utils.get_generative_confidence(output)
        print("=== In get_answer_from_model_output ===")
        # print("Generated sequences shape:", output["sequences"].shape)
        # print("Sample generated sequence (token ids):", output["sequences"][0][:20])
        print("Decoded prediction sample:", pred_answers[0])
        print("Confidence sample:", pred_answers_conf[0] if pred_answers_conf else "None")
        return pred_answers, pred_answers_conf

#############################################
# Example usage
#############################################

# if __name__ == "__main__":
#     config = {
#         'batch_size': 2,
#         'model_weights': 't5-small',
#         'visual_module': {},
#         'max_source_length': 512,
#         'device': 'cuda' if torch.cuda.is_available() else 'cpu'
#     }
    
#     model_instance = VT5_GDOC_FUSION(config)
    
#     batch = {
#         'questions': ["What is in the document?", "Describe the layout."],
#         'words': [
#             ["This", "is", "a", "document"],
#             ["Layout", "analysis", "is", "interesting"]
#         ],
#         'boxes': [
#             [[10, 10, 50, 50]] * 4,
#             [[15, 15, 55, 55]] * 4
#         ],
#         'images': [Image.new('RGB', (224, 224)), Image.new('RGB', (224, 224))],
#         'answers': [["Document", "File"], ["Layout", "Structure"]],
#         # Dummy GraphDoc embeddings and masks:
#         # For example, graphdoc_embeds originally has 289 tokens.
#         'graphdoc_embeds': torch.randn(2, 289, 768),
#         'graphdoc_masks': (torch.rand(2, 289) > 0.5).long()
#     }
    
#     outputs, pred_answers, pred_answer_pages, pred_answers_conf = model_instance.forward(batch)
#     print("Model outputs:", outputs)
