import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5Tokenizer, T5ForConditionalGeneration
import models._model_utils as model_utils
from models._modules import CustomT5Config, SpatialEmbeddings, VisualEmbeddings
import transformers.models.t5.modeling_t5
from PIL import Image
import PIL

###############################################
# Module 1: SequenceProjector (Query-based)
###############################################
class SequenceProjector(nn.Module):
    """
    Projects a variable-length input sequence [B, L_in, d_model] to a fixed-length sequence [B, L_out, d_model]
    using learnable query embeddings and multi-head attention.
    
    For older PyTorch versions (<1.8), we avoid using batch_first.
    """
    def __init__(self, d_model, num_queries, num_heads=8):
        super().__init__()
        self.num_queries = num_queries
        self.d_model = d_model
        # Learnable query embeddings: shape [num_queries, d_model]
        self.queries = nn.Parameter(torch.randn(num_queries, d_model))
        # Create the MultiheadAttention without using batch_first.
        # Note: This module expects input of shape [L, B, d_model]
        self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads)
    
    def forward(self, x, key_padding_mask=None):
        """
        x: Tensor of shape [B, L_in, d_model]
        key_padding_mask: Boolean tensor of shape [B, L_in] with True for positions to ignore.
        Returns: Tensor of shape [B, num_queries, d_model]
        """
        B = x.size(0)
        # Permute x to shape [L_in, B, d_model] for the attention module.
        x_permuted = x.transpose(0, 1)  # [L_in, B, d_model]
        # Expand the queries to shape [num_queries, B, d_model]
        queries = self.queries.unsqueeze(1).expand(-1, B, -1)
        # Compute attention. (No batch_first parameter is needed here.)
        proj_output, _ = self.attention(queries, x_permuted, x_permuted, key_padding_mask=key_padding_mask)
        # proj_output has shape [num_queries, B, d_model]. Transpose back to [B, num_queries, d_model].
        return proj_output.transpose(0, 1)

###############################################
# Module 2: VT5_GDOC_UPSAMPLE_AND_PROJECT
###############################################
class VT5_GDOC_UPSAMPLE_AND_PROJECT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.batch_size = config['batch_size']
        self.tokenizer = T5Tokenizer.from_pretrained(config['model_weights'])
        self.model = T5ForConditionalGeneration.from_pretrained(config['model_weights'])
        
        self.page_retrieval = config['page_retrieval'].lower() if 'page_retrieval' in config else None
        self.max_source_length = config.get('max_source_length', 512)
        
        t5_config = CustomT5Config.from_pretrained(config['model_weights'])
        t5_config.visual_module_config = config['visual_module']
        
        self.spatial_embedding = SpatialEmbeddings(t5_config)
        self.visual_embedding = VisualEmbeddings(t5_config)
        
        # Instead of a token-wise fusion MLP, we now use a learnable sequence projector.
        # After concatenation, the embeddings have shape [B, L_in, 768].
        # We want to map this to a fixed sequence length, e.g. [B, target_seq_len, 768].
        target_seq_len = config.get('target_seq_len', 1024)  # e.g., 1024 tokens
        d_model = 768
        self.sequence_projector = SequenceProjector(d_model=d_model, num_queries=target_seq_len, num_heads=8)
        
        # Move modules to device.
        device = config['device']  # e.g., 'cuda'
        self.spatial_embedding.to(device)
        self.visual_embedding.to(device)
        self.sequence_projector.to(device)
        self.model.to(device)
    
    def parallelize(self):
        self.model = nn.DataParallel(self.model)
    
    def prepare_inputs_for_vqa(self, question, words, boxes, images, answers=None, graphdoc_embeds=None, graphdoc_masks=None):
        bs = len(words)
        prompt_text = ["question: {:s}  context: ".format(q) for q in question]
        prompt_box = [0, 0, 1000, 1000]
        eos_box = [0, 0, 0, 0]
        padding_box_value = 0
        
        longest_seq = 0
        batch_input_ids = []
        batch_input_boxes = []
        for batch_idx in range(bs):
            tokenized_prompt = self.tokenizer(prompt_text[batch_idx])
            input_ids = tokenized_prompt.input_ids[:-1]
            input_boxes = [prompt_box] * len(input_ids)
            
            for word, box in zip(words[batch_idx], boxes[batch_idx]):
                tokenized_word = self.tokenizer(word).input_ids[:-1]
                input_ids.extend(tokenized_word)
                input_boxes.extend([box] * len(tokenized_word))
            
            batch_input_ids.append(input_ids[:self.max_source_length - 1] + [self.tokenizer.eos_token_id])
            batch_input_boxes.append(np.concatenate([input_boxes[:self.max_source_length - 1], np.array([eos_box])]))
            longest_seq = min(max(longest_seq, len(input_ids) + 1), self.max_source_length)
        
        tensor_input_ids = torch.full([bs, longest_seq], fill_value=self.tokenizer.pad_token_id, dtype=torch.long)
        tensor_boxes = torch.full([bs, longest_seq, 4], fill_value=padding_box_value, dtype=torch.long)
        tensor_attention_mask = torch.zeros([bs, longest_seq], dtype=torch.long)
        
        for batch_idx in range(bs):
            tensor_input_ids[batch_idx, :len(batch_input_ids[batch_idx])] = torch.LongTensor(batch_input_ids[batch_idx])
            tensor_boxes[batch_idx, :len(batch_input_boxes[batch_idx])] = torch.from_numpy(batch_input_boxes[batch_idx])
            tensor_attention_mask[batch_idx, :len(batch_input_ids[batch_idx])] = 1
        
        device = self.model.device
        tensor_input_ids = tensor_input_ids.to(device)
        tensor_boxes = tensor_boxes.to(device)
        tensor_attention_mask = tensor_attention_mask.to(device)
        
        # Get semantic embeddings from T5.
        semantic_embedding = self.model.shared(tensor_input_ids)
        # Get spatial embeddings.
        spatial_embedding = self.spatial_embedding(tensor_boxes)
        # Get visual embeddings.
        visual_embedding, visual_emb_mask = self.visual_embedding(images)
        
        # Combine VT5 embeddings: add semantic and spatial, then concatenate visual embeddings.
        vt5_embeds = torch.add(semantic_embedding, spatial_embedding)
        vt5_embeds = torch.cat([vt5_embeds, visual_embedding], dim=1)
        vt5_attention_mask = torch.cat([tensor_attention_mask, visual_emb_mask], dim=1)
        
        # print(f"[DEBUG] VT5 embeddings shape: {vt5_embeds.shape}")
        # print(f"[DEBUG] VT5 attention mask shape: {vt5_attention_mask.shape}")
        
        # Process GraphDoc embeddings if provided.
        if graphdoc_embeds is not None and graphdoc_masks is not None:
            graphdoc_embeds = graphdoc_embeds.to(device)
            graphdoc_masks = graphdoc_masks.to(device)
            # print(f"[DEBUG] GraphDoc embeddings original shape: {graphdoc_embeds.shape}")
            # print(f"[DEBUG] GraphDoc masks shape: {graphdoc_masks.shape}")
            
            target_gdoc_length = 512
            upsampled_gdoc_list = []
            for i in range(graphdoc_embeds.size(0)):
                valid_indices = (graphdoc_masks[i] == 1).nonzero(as_tuple=False).squeeze(1)
                # print(f"[DEBUG] Sample {i}: Number of valid GraphDoc tokens: {valid_indices.numel()}")
                if valid_indices.numel() == 0:
                    valid_embeds = torch.zeros((1, 768), device=device)
                    print(f"[DEBUG] Sample {i}: No valid tokens, using zeros.")
                else:
                    valid_embeds = graphdoc_embeds[i, valid_indices]
                    # print(f"[DEBUG] Sample {i}: valid_embeds shape before upsampling: {valid_embeds.shape}")
                
                valid_embeds = valid_embeds.unsqueeze(0).permute(0, 2, 1)
                upsampled = F.interpolate(valid_embeds, size=target_gdoc_length, mode='linear', align_corners=False)
                upsampled = upsampled.permute(0, 2, 1).squeeze(0)
                # print(f"[DEBUG] Sample {i}: upsampled GraphDoc shape: {upsampled.shape}")
                upsampled_gdoc_list.append(upsampled)
            
            upsampled_gdoc_embeds = torch.stack(upsampled_gdoc_list, dim=0)
            # print(f"[DEBUG] Upsampled GraphDoc embeddings shape: {upsampled_gdoc_embeds.shape}")
            
            gdoc_attention_mask = torch.ones(upsampled_gdoc_embeds.size(0), target_gdoc_length, dtype=torch.long, device=device)
            # print(f"[DEBUG] GraphDoc attention mask shape (after upsampling): {gdoc_attention_mask.shape}")
            
            concatenated_embeds = torch.cat([vt5_embeds, upsampled_gdoc_embeds], dim=1)
            concatenated_attention_mask = torch.cat([vt5_attention_mask, gdoc_attention_mask], dim=1)
            print(f"[DEBUG] Concatenated embeddings shape: {concatenated_embeds.shape}")
            print(f"[DEBUG] Concatenated attention mask shape: {concatenated_attention_mask.shape}")
            
            # Create a key-padding mask for the concatenated embeddings.
            # Assume concatenated_attention_mask has 1 for valid tokens and 0 for padding.
            key_padding_mask = (concatenated_attention_mask == 0)  # True where padding.
            
            # Use the query-based projector to map the variable-length concatenated embeddings to a fixed length.
            projected_embeds = self.sequence_projector(concatenated_embeds, key_padding_mask=key_padding_mask)
            print(f"[DEBUG] Projected embeddings shape (after sequence projector): {projected_embeds.shape}")
            
            # For the new projected output, we assume that all tokens are valid.
            new_attention_mask = torch.ones(projected_embeds.size(0), projected_embeds.size(1), dtype=torch.long, device=device)
            
            input_embeds = projected_embeds
            tensor_attention_mask = new_attention_mask
        else:
            input_embeds = vt5_embeds
            tensor_attention_mask = vt5_attention_mask
        
        # Tokenize answers if provided.
        if answers is not None:
            answers = [random.choice(answer) for answer in answers]
            labels = self.tokenizer(answers, return_tensors='pt', padding=True)
            labels.input_ids[labels.input_ids == self.tokenizer.pad_token_id] = -100
            labels = labels.input_ids.to(device)
        else:
            labels = None
        
        return input_embeds, tensor_attention_mask, labels
    
    def forward(self, batch, return_pred_answer=False):
        question = batch['questions']
        words = batch['words']
        boxes = batch['boxes']
        images = batch['images']
        answers = batch['answers']
        
        graphdoc_embeds = batch.get('graphdoc_embeds')  # [B, gdoc_seq_length, 768]
        graphdoc_masks = batch.get('graphdoc_masks')      # [B, gdoc_seq_length]
        
        bs = len(question)
        
        if self.page_retrieval == 'logits':
            num_pages = batch['num_pages']
            outputs = []
            pred_answers = []
            pred_answer_pages = []
            pred_answers_conf = []
            
            for batch_idx in range(bs):
                input_embeds, attention_mask, _ = self.prepare_inputs_for_vqa(
                    [question[batch_idx]] * num_pages[batch_idx],
                    words[batch_idx],
                    boxes[batch_idx],
                    images,  # assume images can be reused
                    graphdoc_embeds=None,
                    graphdoc_masks=None
                )
                pred_answer, logits = self.get_answer_from_model_output(input_embeds, attention_mask)
                max_logits = -999999
                answer_page = None
                best_answer = None
                for page_ix in range(len(input_embeds)):
                    if logits[page_ix] > max_logits:
                        max_logits = logits[page_ix]
                        answer_page = page_ix
                        best_answer = pred_answer[page_ix]
                
                outputs.append(None)
                pred_answers.append(best_answer)
                pred_answer_pages.append(answer_page)
                pred_answers_conf.append(max_logits)
        else:
            input_embeds, attention_mask, labels = self.prepare_inputs_for_vqa(
                question, words, boxes, images, answers,
                graphdoc_embeds=batch.get('graphdoc_embeds'),
                graphdoc_masks=batch.get('graphdoc_masks')
            )
            outputs = self.model(inputs_embeds=input_embeds, attention_mask=attention_mask, labels=labels)
            pred_answers, pred_answers_conf = (
                self.get_answer_from_model_output(input_embeds, attention_mask)
                if return_pred_answer else (None, None)
            )
            
            if self.page_retrieval == 'oracle':
                pred_answer_pages = batch['answer_page_idx']
            elif self.page_retrieval == 'concat':
                pred_answer_pages = None
            else:
                pred_answer_pages = None
        
        return outputs, pred_answers, pred_answer_pages, pred_answers_conf
    
    def get_answer_from_model_output(self, input_embeds, attention_mask):
        output = self.model.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            output_scores=True,
            return_dict_in_generate=True,
            output_attentions=True
        )
        pred_answers = self.tokenizer.batch_decode(output['sequences'], skip_special_tokens=True)
        pred_answers_conf = model_utils.get_generative_confidence(output)
        print("=== In get_answer_from_model_output ===")
        print("Decoded prediction sample:", pred_answers[0])
        print("Confidence sample:", pred_answers_conf[0] if pred_answers_conf else "None")
        return pred_answers, pred_answers_conf
