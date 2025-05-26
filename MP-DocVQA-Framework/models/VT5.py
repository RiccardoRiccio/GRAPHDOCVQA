'''
THIS WORKS WELL USING T5/BASE WEIGHTS BUT NOT RUBENTITO
add:
 # Move the embedding modules to the same device as the model
        device = config['device']  # typically 'cuda'
        self.spatial_embedding = self.spatial_embedding.to(device)
        self.visual_embedding = self.visual_embedding.to(device)
add:
else:
            pred_answer_pages = None
add:

# Ensure the main model is also on the correct device
self.model = self.model.to(device)

# --- DEBUG: Verify device placement ---
print(f"DEBUG: Model is on device: {next(self.model.parameters()).device}")
print(f"DEBUG: Spatial Embedding is on device: {next(self.spatial_embedding.parameters()).device}")
print(f"DEBUG: Visual Embedding is on device: {next(self.visual_embedding.parameters()).device}")

add:
# --- DEBUG: Check key weight statistics ---
encoder_param_name = "encoder.block.0.layer.0.SelfAttention.q.weight"
decoder_param_name = "decoder.block.0.layer.0.SelfAttention.k.weight"

model_params = dict(self.model.named_parameters())
if encoder_param_name in model_params:
    encoder_param = model_params[encoder_param_name]
    print(f"DEBUG: Encoder '{encoder_param_name}' stats - Mean: {encoder_param.mean().item():.4f}, Std: {encoder_param.std().item():.4f}")
else:
    print(f"DEBUG: Parameter '{encoder_param_name}' not found in the model.")

if decoder_param_name in model_params:
    decoder_param = model_params[decoder_param_name]
    print(f"DEBUG: Decoder '{decoder_param_name}' stats - Mean: {decoder_param.mean().item():.4f}, Std: {decoder_param.std().item():.4f}")
else:
    print(f"DEBUG: Parameter '{decoder_param_name}' not found in the model.")
'''
import random
import numpy as np

import torch
import torch.nn as nn
from click.core import batch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import models._model_utils as model_utils
from models._modules import CustomT5Config, SpatialEmbeddings, VisualEmbeddings
import transformers.models.t5.modeling_t5


class VT5(nn.Module):
    def __init__(self, config):
        super(VT5, self).__init__()
        self.batch_size = config['batch_size']
        self.tokenizer = T5Tokenizer.from_pretrained(config['model_weights'])
        self.model = T5ForConditionalGeneration.from_pretrained(config['model_weights'])

        # --- DEBUG: Check key weight statistics ---
        # encoder_param_name = "encoder.block.0.layer.0.SelfAttention.q.weight"
        # decoder_param_name = "decoder.block.0.layer.0.SelfAttention.k.weight"

        # model_params = dict(self.model.named_parameters())
        # if encoder_param_name in model_params:
        #     encoder_param = model_params[encoder_param_name]
        #     print(f"DEBUG: Encoder '{encoder_param_name}' stats - Mean: {encoder_param.mean().item():.4f}, Std: {encoder_param.std().item():.4f}")
        # else:
        #     print(f"DEBUG: Parameter '{encoder_param_name}' not found in the model.")

        # if decoder_param_name in model_params:
        #     decoder_param = model_params[decoder_param_name]
        #     print(f"DEBUG: Decoder '{decoder_param_name}' stats - Mean: {decoder_param.mean().item():.4f}, Std: {decoder_param.std().item():.4f}")
        # else:
        #     print(f"DEBUG: Parameter '{decoder_param_name}' not found in the model.")
        self.page_retrieval = config['page_retrieval'].lower() if 'page_retrieval' in config else None
        self.max_source_length = config.get('max_source_length', 512)

        t5_config = CustomT5Config.from_pretrained(config['model_weights'])
        t5_config.visual_module_config = config['visual_module']

        self.spatial_embedding = SpatialEmbeddings(t5_config)
        self.visual_embedding = VisualEmbeddings(t5_config)

        # Move the embedding modules to the same device as the model
        device = config['device']  # typically 'cuda'
        self.spatial_embedding = self.spatial_embedding.to(device)
        self.visual_embedding = self.visual_embedding.to(device)


        # Ensure the main model is also on the correct device
        self.model = self.model.to(device)

        # --- DEBUG: Verify device placement ---
        # print(f"DEBUG: Model is on device: {next(self.model.parameters()).device}")
        # print(f"DEBUG: Spatial Embedding is on device: {next(self.spatial_embedding.parameters()).device}")
        # print(f"DEBUG: Visual Embedding is on device: {next(self.visual_embedding.parameters()).device}")

    def parallelize(self):
        self.model = nn.DataParallel(self.model)

    def prepare_inputs_for_vqa(self, question, words, boxes, images, answers=None):
        bs = len(words)
        # input_text = ["question: {:s}  context: {:s}".format(q, c) for q, c in zip(question, words)]
        prompt_text = ["question: {:s}  context: ".format(q) for q in question]
        prompt_box = [0, 0, 1000, 1000]
        eos_box = [0, 0, 0, 0]
        padding_box_value = 0  # To become [0, 0, 0, 0] array.

        # Get input_ids, attention_mask and boxes.
        longest_seq = 0
        batch_input_ids = []
        batch_input_boxes = []
        for batch_idx in range(bs):
            tokenized_prompt = self.tokenizer(prompt_text[batch_idx])
            input_ids = tokenized_prompt.input_ids[:-1]
            input_boxes = [prompt_box] * len(input_ids)

            for word, box in zip(words[batch_idx], boxes[batch_idx]):
                tokenized_word = self.tokenizer(word).input_ids[:-1]  # Tokenize the word and ignore eos_token
                input_ids.extend(tokenized_word)
                scaled_box = [int(round(coord * 1000)) for coord in box]
                input_boxes.extend([scaled_box] * len(tokenized_word))  # Repeat the box for each token corresponding to the word.

            batch_input_ids.append(input_ids[:self.max_source_length-1] + [self.tokenizer.eos_token_id])  # Append the eos_token at the end.
            batch_input_boxes.append(np.concatenate([input_boxes[:self.max_source_length-1],  np.array([eos_box])]))  # Append a bounding box corresponding to the eos_token.
            longest_seq = min(max(longest_seq, len(input_ids) + 1), self.max_source_length)

        # Convert to tensors and pad. Actually, a pad tensor is created and it's filled with corresponding values.
        tensor_input_ids = torch.full([bs, longest_seq], fill_value=self.tokenizer.pad_token_id, dtype=torch.long)
        tensor_boxes = torch.full([bs, longest_seq, 4],  fill_value=padding_box_value, dtype=torch.long)
        tensor_attention_mask = torch.zeros([bs, longest_seq], dtype=torch.long)

        for batch_idx in range(bs):
            tensor_input_ids[batch_idx, :len(batch_input_ids[batch_idx])] = torch.LongTensor(batch_input_ids[batch_idx])
            tensor_boxes[batch_idx, :len(batch_input_boxes[batch_idx])] = torch.from_numpy(batch_input_boxes[batch_idx][:len(batch_input_boxes[batch_idx])])
            tensor_attention_mask[batch_idx, :len(batch_input_ids[batch_idx])] = 1

        """
        context = [(' ').join(doc_words) for doc_words in words]
        input_text = ["question: {:s}  context: {:s}".format(q, c) for q, c in zip(question, context)]
        tokens = self.tokenizer(input_text, return_tensors='pt', padding=True, truncation=True).to(self.model.device)
        input_embeds = self.model.shared(tokens.input_ids)
        """

        # Send everything to GPU
        tensor_input_ids = tensor_input_ids.to(self.model.device)
        tensor_boxes = tensor_boxes.to(self.model.device)
        tensor_attention_mask = tensor_attention_mask.to(self.model.device)

        # Get semantic and spatial embeddings
        semantic_embedding = self.model.shared(tensor_input_ids)
        spatial_embedding = self.spatial_embedding(tensor_boxes)
        visual_embedding, visual_emb_mask = self.visual_embedding(images)

        # input_embeds = semantic_embedding
        input_embeds = torch.add(semantic_embedding, spatial_embedding)
        input_embeds = torch.cat([input_embeds, visual_embedding], dim=1)  # Concatenate semantic + visual embeddings TODO: Provide visual bounding boxes.
        tensor_attention_mask = torch.cat([tensor_attention_mask, visual_emb_mask], dim=1)

         


        # Debug prints to inspect tensor shapes and a small sample of values.
        # print("=== In prepare_inputs_for_vqa ===")
        # print("tensor_input_ids.shape:", tensor_input_ids.shape)
        # print("tensor_boxes.shape:", tensor_boxes.shape)
        # print("semantic_embedding.shape:", semantic_embedding.shape)
        # print("spatial_embedding.shape:", spatial_embedding.shape)
        # print("visual_embedding.shape:", visual_embedding.shape)
        # print("input_embeds.shape:", input_embeds.shape)
        # print("tensor_attention_mask.shape:", tensor_attention_mask.shape)
        # # Optionally print a small sample (first 5 values) from input_embeds:
        # print("Sample input_embeds (first row, first 5 values):", input_embeds[0, :5])

        """
        context = [' '.join(doc_words) for doc_words in words]
        input_text = ["question: {:s}  context: {:s}".format(q, c) for q, c in zip(question, context)]
        tokens = self.tokenizer(input_text, return_tensors='pt', padding=True, truncation=True).to(self.model.device)
        x = self.model.shared(tokens.input_ids)
        """

        # Tokenize answers
        if answers is not None:
            answers = [random.choice(answer) for answer in answers]
            labels = self.tokenizer(answers, return_tensors='pt', padding=True)
            labels.input_ids[labels.input_ids[:] == self.tokenizer.pad_token_id] = -100
            labels = labels.input_ids.to(self.model.device)
        else:
            labels = None

        return input_embeds, tensor_attention_mask, labels

    def forward(self, batch, return_pred_answer=False):
        question = batch['questions']
        words = batch['words']
        boxes = batch['boxes']
        images = batch['images']
        answers = batch['answers']

        # graphdoc_embeds = batch['graphdoc_embeds']  # [B, doc_seq_len, d_model]
        # graphdoc_masks = batch['graphdoc_masks']    # [B, doc_seq_len]


        bs = len(question)

        if self.page_retrieval == 'logits':
            num_pages = batch['num_pages']
            outputs = []
            pred_answers = []
            pred_answer_pages = []
            pred_answers_conf = []

            for batch_idx in range(bs):
                input_embeds, attention_mask, _ = self.prepare_inputs_for_vqa([question[batch_idx]]*num_pages[batch_idx], words[batch_idx], boxes[batch_idx])  # Answers are not considered. Logits set-up is made only for inference.
                pred_answer, logits = self.get_answer_from_model_output(input_embeds, attention_mask)
                # input_text = ["question: {:s}  context: {:s}".format(q, c) for q, c in zip([question[batch_idx]]*len(context[batch_idx]), context[batch_idx])]
                # tokens = self.tokenizer(input_text, return_tensors='pt', padding=True, truncation=True).to(self.model.device)

                max_logits = -999999
                answer_page = None
                best_answer = None
                for page_ix in range(len(input_embeds)):
                    if logits[page_ix] > max_logits:
                        max_logits = logits[page_ix]
                        answer_page = page_ix
                        best_answer = pred_answer[page_ix]

                outputs.append(None)  # outputs.append(document_outputs)  # During inference outputs are not used.
                pred_answers.append(best_answer)
                pred_answer_pages.append(answer_page)
                pred_answers_conf.append(max_logits)

        else:
            input_embeds, attention_mask, labels = self.prepare_inputs_for_vqa(question, words, boxes, images, answers)
            outputs = self.model(inputs_embeds=input_embeds, attention_mask=attention_mask, labels=labels)
            pred_answers, pred_answers_conf = self.get_answer_from_model_output(input_embeds, attention_mask) if return_pred_answer else None

            if self.page_retrieval == 'oracle':
                pred_answer_pages = batch['answer_page_idx']

            elif self.page_retrieval == 'concat':
                pred_answer_pages = None
            else:
                pred_answer_pages = None

        return outputs, pred_answers, pred_answer_pages, pred_answers_conf

    def get_answer_from_model_output(self, input_embeds, attention_mask):
        output = self.model.generate(inputs_embeds=input_embeds, attention_mask=attention_mask, output_scores=True, return_dict_in_generate=True, output_attentions=True)
        pred_answers = self.tokenizer.batch_decode(output['sequences'], skip_special_tokens=True)
        pred_answers_conf = model_utils.get_generative_confidence(output)
         # Debug prints to check model outputs.
        print("=== In get_answer_from_model_output ===")
        # print("Generated sequences shape:", output["sequences"].shape)
        # print("Sample generated sequence (token ids):", output["sequences"][0][:20])
        print("Decoded prediction sample:", pred_answers[0])
        print("Confidence sample:", pred_answers_conf[0] if pred_answers_conf else "None")

        return pred_answers, pred_answers_conf








# #  THIS WORKS WELL USIGN RUBENTITO WEIGHTS BUT NOT T5/BASE
# import random
# import numpy as np

# import torch
# from transformers import T5Tokenizer, T5ForConditionalGeneration
# from transformers import PreTrainedModel
# from models._modules import CustomT5Config, SpatialEmbeddings, VisualEmbeddings
# import models._model_utils as model_utils
# import transformers.models.t5.modeling_t5


# class HF_VT5(PreTrainedModel):

#     def __init__(self, t5_config):
#         super().__init__(t5_config)

#         self.language_backbone = T5ForConditionalGeneration(t5_config)
#         self.spatial_embedding = SpatialEmbeddings(t5_config)
#         self.visual_embedding = VisualEmbeddings(t5_config)


# class VT5:
#     def __init__(self, config):
#         self.max_input_tokens = getattr(config, 'max_input_tokens', 512)

#         t5_config = CustomT5Config.from_pretrained(config['model_weights'])
#         t5_config.visual_module_config = config['visual_module']

#         self.tokenizer = T5Tokenizer.from_pretrained(config['model_weights'])

#         self.model = HF_VT5.from_pretrained(config['model_weights'], config=t5_config)

#         # Initialize page_retrieval attribute
#         self.page_retrieval = getattr(config, 'page_retrieval', None)
        
#          # Expose submodules explicitly
#         self.spatial_embedding = self.model.spatial_embedding
#         self.visual_embedding = self.model.visual_embedding
#         self.language_backbone = self.model.language_backbone


#     def prepare_inputs_for_vqa(self, question, words, boxes, images, answers=None):
#         bs = len(words)
#         # input_text = ["question: {:s}  context: {:s}".format(q, c) for q, c in zip(question, words)]
#         prompt_text = ["question: {:s}  context: ".format(q) for q in question]
#         prompt_box = [0, 0, 1000, 1000]
#         eos_box = [0, 0, 0, 0]
#         padding_box_value = 0  # To become [0, 0, 0, 0] array.

#         # Get input_ids, attention_mask and boxes.
#         longest_seq = 0
#         batch_input_ids = []
#         batch_input_boxes = []
#         for batch_idx in range(bs):
#             tokenized_prompt = self.tokenizer(prompt_text[batch_idx])
#             input_ids = tokenized_prompt.input_ids[:-1]
#             input_boxes = [prompt_box] * len(input_ids)

#             for word, box in zip(words[batch_idx], boxes[batch_idx]):
#                 tokenized_word = self.tokenizer(word).input_ids[:-1]  # Tokenize the word and ignore eos_token
#                 input_ids.extend(tokenized_word)
#                 input_boxes.extend([box]*len(tokenized_word))  # Repeat the box for each token corresponding to the word.

#             batch_input_ids.append(input_ids[:self.max_input_tokens-1] + [self.tokenizer.eos_token_id])  # Append the eos_token at the end.
#             batch_input_boxes.append(np.concatenate([input_boxes[:self.max_input_tokens-1],  np.array([eos_box])]))  # Append a bounding box corresponding to the eos_token.
#             longest_seq = min(max(longest_seq, len(input_ids) + 1), self.max_input_tokens)

#         # Convert to tensors and pad. Actually, a pad tensor is created and it's filled with corresponding values.
#         tensor_input_ids = torch.full([bs, longest_seq], fill_value=self.tokenizer.pad_token_id, dtype=torch.long)
#         tensor_boxes = torch.full([bs, longest_seq, 4],  fill_value=padding_box_value, dtype=torch.long)
#         tensor_attention_mask = torch.zeros([bs, longest_seq], dtype=torch.long)

#         for batch_idx in range(bs):
#             tensor_input_ids[batch_idx, :len(batch_input_ids[batch_idx])] = torch.LongTensor(batch_input_ids[batch_idx])
#             tensor_boxes[batch_idx, :len(batch_input_boxes[batch_idx])] = torch.from_numpy(batch_input_boxes[batch_idx][:len(batch_input_boxes[batch_idx])])
#             tensor_attention_mask[batch_idx, :len(batch_input_ids[batch_idx])] = 1

#         # Send everything to GPU
#         tensor_input_ids = tensor_input_ids.to(self.model.device)
#         tensor_boxes = tensor_boxes.to(self.model.device)
#         tensor_attention_mask = tensor_attention_mask.to(self.model.device)

#         # Get semantic and spatial embeddings
#         semantic_embedding = self.model.language_backbone.shared(tensor_input_ids)
#         spatial_embedding = self.model.spatial_embedding(tensor_boxes)
#         visual_embedding, visual_emb_mask = self.model.visual_embedding(images)

#         # input_embeds = semantic_embedding
#         input_embeds = torch.add(semantic_embedding, spatial_embedding)
#         input_embeds = torch.cat([input_embeds, visual_embedding], dim=1)  # Concatenate semantic + visual embeddings TODO: Provide visual bounding boxes.
#         tensor_attention_mask = torch.cat([tensor_attention_mask, visual_emb_mask], dim=1)

#         # Tokenize answers
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

#         input_embeds, attention_mask, labels = self.prepare_inputs_for_vqa(question, words, boxes, images, answers)
#         outputs = self.model.language_backbone(inputs_embeds=input_embeds, attention_mask=attention_mask, labels=labels)
#         pred_answers, pred_answers_conf = self.get_answer_from_model_output(input_embeds, attention_mask) if return_pred_answer else None

#         if self.page_retrieval == 'oracle':
#             pred_answer_page = batch['answer_page_idx']
#         elif self.page_retrieval == 'concat':
#             pred_answer_page = None
#         else:
#             pred_answer_page = None  # Add a default value for pred_answer_page

#         return outputs, pred_answers, pred_answer_page, pred_answers_conf

      

#     def get_answer_from_model_output(self, input_embeds, attention_mask):
#         output = self.model.language_backbone.generate(inputs_embeds=input_embeds, attention_mask=attention_mask, output_scores=True, return_dict_in_generate=True, output_attentions=True)
#         pred_answers = self.tokenizer.batch_decode(output['sequences'], skip_special_tokens=True)
#         pred_answers_conf = model_utils.get_generative_confidence(output)

#         return pred_answers, pred_answers_conf


