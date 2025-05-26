# """
# Revised VT5_GDOC model that adds modality-specific embeddings for text, visual, 
# and GraphDoc tokens. The idea is to mitigate the text bias inherent in a T5 encoder–decoder 
# by informing it which tokens come from which modality.
# """

# import random
# import numpy as np
# import torch
# import torch.nn as nn
# from transformers import T5Tokenizer, T5ForConditionalGeneration
# import models._model_utils as model_utils
# from models._modules import CustomT5Config, SpatialEmbeddings, VisualEmbeddings
# from PIL import Image
# import PIL


# class VT5_GDOC_MULTIMODAL:
#     def __init__(self, config):
#         print("USING MULTIMODAL MODEL")
#         self.batch_size = config['batch_size']
#         self.tokenizer = T5Tokenizer.from_pretrained(config['model_weights'])
#         self.model = T5ForConditionalGeneration.from_pretrained(config['model_weights'])
        
#         # Page retrieval option (if used)
#         self.page_retrieval = config['page_retrieval'].lower() if 'page_retrieval' in config else None
#         self.max_source_length = config.get('max_source_length', 512)
        
#         # Load a custom T5 config and pass visual module configuration to it.
#         t5_config = CustomT5Config.from_pretrained(config['model_weights'])
#         t5_config.visual_module_config = config['visual_module']
        
#         # Initialize spatial and visual embeddings modules.
#         self.spatial_embedding = SpatialEmbeddings(t5_config)
#         self.visual_embedding = VisualEmbeddings(t5_config)
        
#         # Move modules to the specified device.
#         device = config['device']
#         self.spatial_embedding = self.spatial_embedding.to(device)
#         self.visual_embedding = self.visual_embedding.to(device)
#         self.model = self.model.to(device)
        
#         # ---------------------------------------------------------------------
#         # Introduce learnable modality embeddings to inform the model which tokens
#         # come from text (OCR + spatial), visual (image patches), or GraphDoc.
#         # These embeddings have the same dimension as the model hidden size (e.g., 768).
#         # ---------------------------------------------------------------------
#         self.text_modality_emb = nn.Parameter(torch.randn(1, 1, self.model.config.d_model))
#         self.visual_modality_emb = nn.Parameter(torch.randn(1, 1, self.model.config.d_model))
#         self.graphdoc_modality_emb = nn.Parameter(torch.randn(1, 1, self.model.config.d_model))
        
#         # Move the modality embeddings to the same device.
#         self.text_modality_emb = self.text_modality_emb.to(device)
#         self.visual_modality_emb = self.visual_modality_emb.to(device)
#         self.graphdoc_modality_emb = self.graphdoc_modality_emb.to(device)
    
#     def parallelize(self):
#         self.model = nn.DataParallel(self.model)
    
#     def prepare_inputs_for_vqa(self, question, words, boxes, images, answers=None, graphdoc_embeds=None, graphdoc_masks=None):
#         bs = len(words)
#         prompt_text = ["question: {:s}  context: ".format(q) for q in question]
#         prompt_box = [0, 0, 1000, 1000]
#         eos_box = [0, 0, 0, 0]
#         padding_box_value = 0  # Will become [0, 0, 0, 0]
        
#         # Prepare token ids and bounding boxes for text tokens.
#         longest_seq = 0
#         batch_input_ids = []
#         batch_input_boxes = []
#         for batch_idx in range(bs):
#             tokenized_prompt = self.tokenizer(prompt_text[batch_idx])
#             input_ids = tokenized_prompt.input_ids[:-1]  # Remove the EOS token
#             input_boxes = [prompt_box] * len(input_ids)
#             for word, box in zip(words[batch_idx], boxes[batch_idx]):
#                 tokenized_word = self.tokenizer(word).input_ids[:-1]
#                 input_ids.extend(tokenized_word)
#                 input_boxes.extend([box] * len(tokenized_word))
#             # Truncate and append the EOS token.
#             batch_input_ids.append(input_ids[:self.max_source_length-1] + [self.tokenizer.eos_token_id])
#             batch_input_boxes.append(np.concatenate([np.array(input_boxes[:self.max_source_length-1]), np.array([eos_box])]))
#             longest_seq = min(max(longest_seq, len(input_ids) + 1), self.max_source_length)
        
#         # Create padded tensors for input ids, bounding boxes, and attention mask.
#         tensor_input_ids = torch.full([bs, longest_seq], fill_value=self.tokenizer.pad_token_id, dtype=torch.long)
#         tensor_boxes = torch.full([bs, longest_seq, 4], fill_value=padding_box_value, dtype=torch.long)
#         tensor_attention_mask = torch.zeros([bs, longest_seq], dtype=torch.long)
        
#         for batch_idx in range(bs):
#             tensor_input_ids[batch_idx, :len(batch_input_ids[batch_idx])] = torch.LongTensor(batch_input_ids[batch_idx])
#             tensor_boxes[batch_idx, :len(batch_input_boxes[batch_idx])] = torch.from_numpy(batch_input_boxes[batch_idx])
#             tensor_attention_mask[batch_idx, :len(batch_input_ids[batch_idx])] = 1
        
#         # Move tensors to the model's device.
#         tensor_input_ids = tensor_input_ids.to(self.model.device)
#         tensor_boxes = tensor_boxes.to(self.model.device)
#         tensor_attention_mask = tensor_attention_mask.to(self.model.device)
        
#         # ------------------------------
#         # Compute the text (semantic + spatial) embeddings.
#         # ------------------------------
#         semantic_embedding = self.model.shared(tensor_input_ids)
#         spatial_embedding = self.spatial_embedding(tensor_boxes)
#         # Sum the text embeddings and add a modality marker.
#         text_embeddings = semantic_embedding + spatial_embedding + self.text_modality_emb
        
#         # ------------------------------
#         # Compute the visual embeddings.
#         # ------------------------------
#         visual_embedding, visual_emb_mask = self.visual_embedding(images)
#         # Add the modality embedding for visual tokens.
#         visual_embeddings = visual_embedding + self.visual_modality_emb
        
#         # Concatenate text and visual tokens.
#         input_embeds = torch.cat([text_embeddings, visual_embeddings], dim=1)
#         tensor_attention_mask = torch.cat([tensor_attention_mask, visual_emb_mask], dim=1)
        
#         # ------------------------------
#         # Optionally, if GraphDoc embeddings are provided, add them.
#         # ------------------------------
#         if graphdoc_embeds is not None and graphdoc_masks is not None:
#             graphdoc_embeds = graphdoc_embeds.to(input_embeds.device)
#             graphdoc_masks = graphdoc_masks.to(tensor_attention_mask.device)
#             # Add modality embedding for GraphDoc tokens.
#             graphdoc_embeddings = graphdoc_embeds + self.graphdoc_modality_emb
#             input_embeds = torch.cat([input_embeds, graphdoc_embeddings], dim=1)
#             tensor_attention_mask = torch.cat([tensor_attention_mask, graphdoc_masks], dim=1)
        
#         # ------------------------------
#         # Tokenize answers (if provided) and prepare labels.
#         # ------------------------------
#         if answers is not None:
#             answers = [random.choice(answer) for answer in answers]
#             labels = self.tokenizer(answers, return_tensors='pt', padding=True)
#             labels.input_ids[labels.input_ids[:] == self.tokenizer.pad_token_id] = -100
#             labels = labels.input_ids.to(self.model.device)
#         else:
#             labels = None
        
#         return input_embeds, tensor_attention_mask, labels
    
#     def forward(self, batch, return_pred_answer=False):
#         question = batch['questions']
#         words = batch['words']
#         boxes = batch['boxes']
#         images = batch['images']
#         answers = batch['answers']
        
#         graphdoc_embeds = batch.get('graphdoc_embeds')
#         graphdoc_masks = batch.get('graphdoc_masks')
        
#         bs = len(question)
        
#         if self.page_retrieval == 'logits':
#             # This branch is used for page retrieval using logits.
#             num_pages = batch['num_pages']
#             outputs = []
#             pred_answers = []
#             pred_answer_pages = []
#             pred_answers_conf = []
            
#             for batch_idx in range(bs):
#                 # Prepare inputs for each page (for inference only).
#                 input_embeds, attention_mask, _ = self.prepare_inputs_for_vqa(
#                     [question[batch_idx]] * num_pages[batch_idx],
#                     words[batch_idx],
#                     boxes[batch_idx],
#                     images[batch_idx]
#                 )
#                 pred_answer, logits = self.get_answer_from_model_output(input_embeds, attention_mask)
#                 max_logits = -999999
#                 answer_page = None
#                 best_answer = None
#                 for page_ix in range(len(input_embeds)):
#                     if logits[page_ix] > max_logits:
#                         max_logits = logits[page_ix]
#                         answer_page = page_ix
#                         best_answer = pred_answer[page_ix]
#                 outputs.append(None)
#                 pred_answers.append(best_answer)
#                 pred_answer_pages.append(answer_page)
#                 pred_answers_conf.append(max_logits)
#         else:
#             # Standard forward pass (for training or standard inference)
#             input_embeds, attention_mask, labels = self.prepare_inputs_for_vqa(
#                 question, words, boxes, images, answers,
#                 graphdoc_embeds=batch.get('graphdoc_embeds'),
#                 graphdoc_masks=batch.get('graphdoc_masks')
#             )
#             outputs = self.model(inputs_embeds=input_embeds, attention_mask=attention_mask, labels=labels)
#             pred_answers, pred_answers_conf = (
#                 self.get_answer_from_model_output(input_embeds, attention_mask)
#                 if return_pred_answer
#                 else (None, None)
#             )
#             if self.page_retrieval == 'oracle':
#                 pred_answer_pages = batch['answer_page_idx']
#             else:
#                 pred_answer_pages = None
        
#         return outputs, pred_answers, pred_answer_pages, pred_answers_conf
    
#     def get_answer_from_model_output(self, input_embeds, attention_mask):
#         output = self.model.generate(
#             inputs_embeds=input_embeds,
#             attention_mask=attention_mask,
#             output_scores=True,
#             return_dict_in_generate=True,
#             output_attentions=True
#         )
#         pred_answers = self.tokenizer.batch_decode(output['sequences'], skip_special_tokens=True)
#         pred_answers_conf = model_utils.get_generative_confidence(output)
#         # Debug prints
#         print("=== In get_answer_from_model_output ===")
#         print("Decoded prediction sample:", pred_answers[0])
#         print("Confidence sample:", pred_answers_conf[0] if pred_answers_conf else "None")
#         return pred_answers, pred_answers_conf

"""
Innovative VT5_GDOC model that fuses text, visual, and GraphDoc precomputed embeddings.
This version uses modality-specific embeddings and a GraphDoc fusion layer that employs 
multi-head cross-attention with a gating mechanism. The fusion layer has been adapted 
to work with PyTorch versions less than 1.8 (i.e. without the 'batch_first' argument).
"""

import random
import numpy as np
import torch
import torch.nn as nn
from transformers import T5Tokenizer, T5ForConditionalGeneration
import models._model_utils as model_utils
from models._modules import CustomT5Config, SpatialEmbeddings, VisualEmbeddings
from PIL import Image
import PIL

# ----------------------------------------------------------------------
# GraphDocFusionLayer: Fuses GraphDoc tokens into the text–visual tokens.
# It uses multi-head attention (without batch_first) by transposing the inputs.
# ----------------------------------------------------------------------
class GraphDocFusionLayer(nn.Module):
    def __init__(self, d_model, nhead=8):
        super(GraphDocFusionLayer, self).__init__()
        # Note: PyTorch < 1.8 does not support batch_first, so we omit that argument.
        self.multihead_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead)
        self.gate_linear = nn.Linear(d_model, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(self, text_visual, graphdoc):
        """
        Args:
            text_visual: Tensor of shape [B, L, d_model] (combined text+visual tokens)
            graphdoc: Tensor of shape [B, L_g, d_model] (GraphDoc tokens)
        Returns:
            Fused tensor of shape [B, L, d_model]
        """
        # Transpose inputs to [L, B, d_model] as required by nn.MultiheadAttention.
        text_visual_t = text_visual.transpose(0, 1)  # [L, B, d_model]
        graphdoc_t = graphdoc.transpose(0, 1)          # [L_g, B, d_model]
        attn_output, _ = self.multihead_attn(query=text_visual_t,
                                             key=graphdoc_t,
                                             value=graphdoc_t)
        # Transpose the attention output back to [B, L, d_model].
        attn_output = attn_output.transpose(0, 1)
        # Compute a gating factor (values between 0 and 1) for each token.
        gate = torch.sigmoid(self.gate_linear(text_visual))
        # Fuse the original text-visual tokens with the attended GraphDoc tokens.
        fused = text_visual + gate * attn_output
        fused = self.layer_norm(fused)
        return fused

# ----------------------------------------------------------------------
# Main VT5_GDOC_MULTIMODAL model definition.
# ----------------------------------------------------------------------
class VT5_GDOC_MULTIMODAL(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.batch_size = config['batch_size']
        self.tokenizer = T5Tokenizer.from_pretrained(config['model_weights'])
        self.model = T5ForConditionalGeneration.from_pretrained(config['model_weights'])
        
        self.page_retrieval = config['page_retrieval'].lower() if 'page_retrieval' in config else None
        self.max_source_length = config.get('max_source_length', 512)
        
        # Load a custom T5 configuration and include visual module settings.
        t5_config = CustomT5Config.from_pretrained(config['model_weights'])
        t5_config.visual_module_config = config['visual_module']
        
        # Initialize spatial and visual embeddings.
        self.spatial_embedding = SpatialEmbeddings(t5_config)
        self.visual_embedding = VisualEmbeddings(t5_config)
        
        device = config['device']
        self.spatial_embedding = self.spatial_embedding.to(device)
        self.visual_embedding = self.visual_embedding.to(device)
        self.model = self.model.to(device)
        
        # ------------------------------------------------------------------
        # Define modality-specific learnable embeddings.
        # They help the model distinguish tokens coming from text, images, or GraphDoc.
        # ------------------------------------------------------------------
        # Define modality-specific learnable embeddings and initialize them directly on the correct device
        self.text_modality_emb = nn.Parameter(torch.randn(1, 1, self.model.config.d_model, device=device))
        self.visual_modality_emb = nn.Parameter(torch.randn(1, 1, self.model.config.d_model, device=device))
        self.graphdoc_modality_emb = nn.Parameter(torch.randn(1, 1, self.model.config.d_model, device=device))

        
        # ------------------------------------------------------------------
        # Initialize the GraphDoc fusion module.
        # ------------------------------------------------------------------
        self.graphdoc_fusion = GraphDocFusionLayer(d_model=self.model.config.d_model, nhead=8).to(device)
    
    def parallelize(self):
        self.model = nn.DataParallel(self.model)
    
    def prepare_inputs_for_vqa(self, question, words, boxes, images, answers=None, graphdoc_embeds=None, graphdoc_masks=None):
        bs = len(words)
        prompt_text = ["question: {:s}  context: ".format(q) for q in question]
        prompt_box = [0, 0, 1000, 1000]
        eos_box = [0, 0, 0, 0]
        padding_box_value = 0
        
        # Build token IDs and bounding boxes for OCR tokens.
        longest_seq = 0
        batch_input_ids = []
        batch_input_boxes = []
        for batch_idx in range(bs):
            tokenized_prompt = self.tokenizer(prompt_text[batch_idx])
            input_ids = tokenized_prompt.input_ids[:-1]  # Remove EOS from prompt.
            input_boxes = [prompt_box] * len(input_ids)
            for word, box in zip(words[batch_idx], boxes[batch_idx]):
                tokenized_word = self.tokenizer(word).input_ids[:-1]
                input_ids.extend(tokenized_word)
                input_boxes.extend([box] * len(tokenized_word))
            # Truncate if necessary and append EOS.
            batch_input_ids.append(input_ids[:self.max_source_length-1] + [self.tokenizer.eos_token_id])
            batch_input_boxes.append(np.concatenate([np.array(input_boxes[:self.max_source_length-1]), np.array([eos_box])]))
            longest_seq = min(max(longest_seq, len(input_ids) + 1), self.max_source_length)
        
        # Create padded tensors.
        tensor_input_ids = torch.full([bs, longest_seq], fill_value=self.tokenizer.pad_token_id, dtype=torch.long)
        tensor_boxes = torch.full([bs, longest_seq, 4], fill_value=padding_box_value, dtype=torch.long)
        tensor_attention_mask = torch.zeros([bs, longest_seq], dtype=torch.long)
        
        for batch_idx in range(bs):
            tensor_input_ids[batch_idx, :len(batch_input_ids[batch_idx])] = torch.LongTensor(batch_input_ids[batch_idx])
            tensor_boxes[batch_idx, :len(batch_input_boxes[batch_idx])] = torch.from_numpy(batch_input_boxes[batch_idx])
            tensor_attention_mask[batch_idx, :len(batch_input_ids[batch_idx])] = 1
        
        # Move tensors to the model's device.
        tensor_input_ids = tensor_input_ids.to(self.model.device)
        tensor_boxes = tensor_boxes.to(self.model.device)
        tensor_attention_mask = tensor_attention_mask.to(self.model.device)
        
        # ------------------------------------------------------------------
        # Compute text embeddings: sum semantic and spatial embeddings, then add a text modality marker.
        # ------------------------------------------------------------------
        semantic_embedding = self.model.shared(tensor_input_ids)
        spatial_embedding = self.spatial_embedding(tensor_boxes)
        text_embeddings = semantic_embedding + spatial_embedding + self.text_modality_emb
        
        # ------------------------------------------------------------------
        # Compute visual embeddings and add the visual modality marker.
        # ------------------------------------------------------------------
        visual_embedding, visual_emb_mask = self.visual_embedding(images)
        visual_embeddings = visual_embedding + self.visual_modality_emb
        
        # Concatenate text and visual tokens.
        base_embeddings = torch.cat([text_embeddings, visual_embeddings], dim=1)
        base_attention_mask = torch.cat([tensor_attention_mask, visual_emb_mask], dim=1)
        
        # ------------------------------------------------------------------
        # Fuse GraphDoc embeddings (if provided) into the base embeddings.
        # ------------------------------------------------------------------
        if graphdoc_embeds is not None and graphdoc_masks is not None:
            graphdoc_embeds = graphdoc_embeds.to(base_embeddings.device) + self.graphdoc_modality_emb
            fused_embeddings = self.graphdoc_fusion(base_embeddings, graphdoc_embeds)
            input_embeds = fused_embeddings
            final_attention_mask = base_attention_mask
        else:
            input_embeds = base_embeddings
            final_attention_mask = base_attention_mask
        
        # ------------------------------------------------------------------
        # Prepare answer labels (if answers are provided).
        # ------------------------------------------------------------------
        if answers is not None:
            answers = [random.choice(answer) for answer in answers]
            labels = self.tokenizer(answers, return_tensors='pt', padding=True)
            labels.input_ids[labels.input_ids[:] == self.tokenizer.pad_token_id] = -100
            labels = labels.input_ids.to(self.model.device)
        else:
            labels = None
        
        return input_embeds, final_attention_mask, labels
    
    def forward(self, batch, return_pred_answer=False):
        question = batch['questions']
        words = batch['words']
        boxes = batch['boxes']
        images = batch['images']
        answers = batch['answers']
        
        graphdoc_embeds = batch.get('graphdoc_embeds')
        graphdoc_masks = batch.get('graphdoc_masks')
        
        bs = len(question)
        
        if self.page_retrieval == 'logits':
            # Branch for page retrieval using logits.
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
                    images[batch_idx]
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
            # Standard forward pass.
            input_embeds, attention_mask, labels = self.prepare_inputs_for_vqa(
                question, words, boxes, images, answers,
                graphdoc_embeds=batch.get('graphdoc_embeds'),
                graphdoc_masks=batch.get('graphdoc_masks')
            )
            outputs = self.model(inputs_embeds=input_embeds, attention_mask=attention_mask, labels=labels)
            if return_pred_answer:
                pred_answers, pred_answers_conf = self.get_answer_from_model_output(input_embeds, attention_mask)
            else:
                pred_answers, pred_answers_conf = None, None
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
            return_dict_in_generate=True,
            output_attentions=True
        )
        pred_answers = self.tokenizer.batch_decode(output['sequences'], skip_special_tokens=True)
        pred_answers_conf = model_utils.get_generative_confidence(output)
        # Debug prints.
        print("=== In get_answer_from_model_output ===")
        print("Decoded prediction sample:", pred_answers[0])
        print("Confidence sample:", pred_answers_conf[0] if pred_answers_conf else "None")
        return pred_answers, pred_answers_conf
