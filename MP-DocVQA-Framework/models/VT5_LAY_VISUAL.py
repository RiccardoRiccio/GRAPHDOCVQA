
















import random
import numpy as np

import torch
import torch.nn as nn
from transformers import T5Tokenizer, T5ForConditionalGeneration
import models._model_utils as model_utils
from models._modules import CustomT5Config, SpatialEmbeddings, VisualEmbeddings
import transformers.models.t5.modeling_t5


class VT5_LAY_VISUAL(nn.Module):
    def __init__(self, config):
        super(VT5_LAY_VISUAL, self).__init__()
        print("USING VT5 LAY GDOC")
        self.batch_size = config['batch_size']
        self.tokenizer = T5Tokenizer.from_pretrained(config['model_weights'])
        self.model = T5ForConditionalGeneration.from_pretrained(config['model_weights'])
        self.page_retrieval = config['page_retrieval'].lower() if 'page_retrieval' in config else None
        self.max_source_length = config.get('max_source_length', 512)

        t5_config = CustomT5Config.from_pretrained(config['model_weights'])
        t5_config.visual_module_config = config['visual_module']

        self.spatial_embedding = SpatialEmbeddings(t5_config)
        self.visual_embedding = VisualEmbeddings(t5_config)

        # --- NEW: Instantiate the extra entity embeddings.
        # We set the number of entity classes to 12:
        # 0–9: detected classes,
        # 10: no entity (or non-overlap),
        # 11: question/prompt tokens.
        self.num_entity_classes = 12
        self.entity_tag_embedding = nn.Embedding(self.num_entity_classes, t5_config.hidden_size)
        # We use the same SpatialEmbeddings class to obtain spatial embeddings for entity boxes.
        self.entity_spatial_embedding = SpatialEmbeddings(t5_config)

        device = config['device']
        
        self.spatial_embedding.to(device)
        self.visual_embedding.to(device)
        self.entity_tag_embedding.to(device)
        self.entity_spatial_embedding.to(device)
        self.model.to(device)


    def parallelize(self):
        self.model = nn.DataParallel(self.model)

    def prepare_inputs_for_vqa(self, question, words, boxes, images, answers=None,
                               entity_tags=None, entity_boxes=None):
        """
        Prepares the inputs for VQA.
        In addition to token semantic and spatial embeddings (as before),
        if entity information is provided (entity_tags and entity_boxes) they are also embedded
        and added to the token representation.
        If not provided, default values are used.
        """
        bs = len(words)
        # Create prompt text (one per question)
        prompt_text = ["question: {:s}  context: ".format(q) for q in question]
        # Use floats for the new entity parts (normalized coordinates)
        prompt_box = [0.0, 0.0, 1000.0, 1000.0]
        eos_box = [0.0, 0.0, 0.0, 0.0]
        padding_box_value = 0

        # If no entity information is given, use defaults.
        # (For OCR tokens, a tag of -1 is later remapped to 10: “no entity”.)
        if entity_tags is None:
            entity_tags = [[-1] * len(doc_words) for doc_words in words]
        if entity_boxes is None:
            entity_boxes = [[[0.0, 0.0, 0.0, 0.0] for _ in doc_words] for doc_words in words]

        longest_seq = 0
        batch_input_ids = []
        batch_input_boxes = []
        batch_entity_tags = []
        batch_entity_boxes = []

        for batch_idx in range(bs):
            # Tokenize the prompt text.
            tokenized_prompt = self.tokenizer(prompt_text[batch_idx])
            input_ids = tokenized_prompt.input_ids[:-1]  # remove prompt EOS token
            input_boxes = [prompt_box] * len(input_ids)
            # For prompt tokens, assign a special entity tag (11) and use eos_box for entity bbox.
            etags = [11] * len(input_ids)
            eboxes = [prompt_box] * len(input_ids)

            # Process each OCR word along with its token box and entity information.
            for word, box, etag, ebox in zip(words[batch_idx], boxes[batch_idx],
                                             entity_tags[batch_idx], entity_boxes[batch_idx]):
                tokenized_word = self.tokenizer(word).input_ids[:-1]  # ignore EOS from tokenization
                input_ids.extend(tokenized_word)
                input_boxes.extend([box] * len(tokenized_word))
                # Remap a tag of -1 to 10 (meaning “no entity”)
                tag_val = etag if etag != -1 else 10
                etags.extend([tag_val] * len(tokenized_word))
                eboxes.extend([ebox] * len(tokenized_word))

            # Truncate to max_source_length - 1 and then append the EOS token.
            truncated_ids = input_ids[:self.max_source_length - 1]
            batch_input_ids.append(truncated_ids + [self.tokenizer.eos_token_id])
            truncated_boxes = np.array(input_boxes[:self.max_source_length - 1], dtype=np.float32)
            batch_input_boxes.append(np.concatenate([truncated_boxes, np.array([eos_box], dtype=np.float32)]))
            truncated_etags = etags[:self.max_source_length - 1]
            # For the EOS token, we assign tag 10 (i.e. “no entity”)
            batch_entity_tags.append(truncated_etags + [10])
            truncated_eboxes = np.array(eboxes[:self.max_source_length - 1], dtype=np.float32)
            batch_entity_boxes.append(np.concatenate([truncated_eboxes, np.array([eos_box], dtype=np.float32)]))

            longest_seq = min(max(longest_seq, len(truncated_ids) + 1), self.max_source_length)

        # Convert lists to padded tensors.
        tensor_input_ids = torch.full([bs, longest_seq],
                                      fill_value=self.tokenizer.pad_token_id, dtype=torch.long)
        tensor_boxes = torch.full([bs, longest_seq, 4],
                                  fill_value=padding_box_value, dtype=torch.float32)
        tensor_entity_tags = torch.full([bs, longest_seq],
                                        fill_value=10, dtype=torch.long)  # default: no entity
        tensor_entity_boxes = torch.full([bs, longest_seq, 4],
                                         fill_value=padding_box_value, dtype=torch.float32)
        tensor_attention_mask = torch.zeros([bs, longest_seq], dtype=torch.long)

        for batch_idx in range(bs):
            cur_len = len(batch_input_ids[batch_idx])
            tensor_input_ids[batch_idx, :cur_len] = torch.LongTensor(batch_input_ids[batch_idx])
            tensor_boxes[batch_idx, :cur_len] = torch.from_numpy(batch_input_boxes[batch_idx])
            tensor_entity_tags[batch_idx, :cur_len] = torch.LongTensor(batch_entity_tags[batch_idx])
            tensor_entity_boxes[batch_idx, :cur_len] = torch.from_numpy(batch_entity_boxes[batch_idx])
            tensor_attention_mask[batch_idx, :cur_len] = 1

        # Send tensors to the same device as the model.
        tensor_input_ids = tensor_input_ids.to(self.model.device)
        tensor_boxes = tensor_boxes.to(self.model.device)
        tensor_entity_tags = tensor_entity_tags.to(self.model.device)
        tensor_entity_boxes = tensor_entity_boxes.to(self.model.device)
        tensor_attention_mask = tensor_attention_mask.to(self.model.device)

        # Scale box coordinates from normalized floats ([0, 1] or similar) to [0, 1000]
        # and convert to integers for the spatial embedding modules.
        # Convert tensor_boxes to Long, then selectively scale OCR tokens.
        scaled_tensor_boxes = tensor_boxes.clone().long()
        mask = (tensor_entity_tags != 11)  # True for OCR tokens, False for prompt tokens
        scaled_tensor_boxes[mask] = (tensor_boxes[mask] * 1000).long()
        scaled_tensor_boxes[~mask] = tensor_boxes[~mask].long()

        # For entity boxes (only OCR tokens), scale them all:
        # For entity boxes, conditionally scale OCR tokens while leaving question tokens unchanged.
        scaled_tensor_entity_boxes = tensor_entity_boxes.clone().long()
        mask_entity = (tensor_entity_tags != 11)  # OCR tokens: tag != 11; prompt tokens: tag == 11.
        scaled_tensor_entity_boxes[mask_entity] = (tensor_entity_boxes[mask_entity] * 1000).long()
        scaled_tensor_entity_boxes[~mask_entity] = tensor_entity_boxes[~mask_entity].long()



        # print("scaled_tensor_boxes", scaled_tensor_boxes)
        # print("scaled_tensor_boxes shape", scaled_tensor_boxes.shape)
        
        # print(" scaled_tensor_entity_boxes",  scaled_tensor_entity_boxes)
        # print(" scaled_tensor_entity_boxes shape:",  scaled_tensor_entity_boxes.shape)

        # Get semantic embeddings from the shared word embeddings.
        semantic_embedding = self.model.shared(tensor_input_ids)
        # Get token-level spatial embeddings.
        spatial_embedding = self.spatial_embedding(scaled_tensor_boxes)
        # Get entity tag embeddings.
        entity_tag_emb = self.entity_tag_embedding(tensor_entity_tags)
        # Get entity spatial embeddings.
        entity_spatial_emb = self.entity_spatial_embedding(scaled_tensor_entity_boxes)
        # Create a mask: tokens that are NOT in an entity (tag 10) get zeroed out.
        mask = (tensor_entity_tags != 10).unsqueeze(-1).float()
        entity_tag_emb = entity_tag_emb * mask
        entity_spatial_emb = entity_spatial_emb * mask

        # Sum all four embedding components.
        input_embeds = semantic_embedding + spatial_embedding + entity_tag_emb + entity_spatial_emb

        # Concatenate visual embeddings.
        visual_embedding, visual_emb_mask = self.visual_embedding(images)
        input_embeds = torch.cat([input_embeds, visual_embedding], dim=1)
        tensor_attention_mask = torch.cat([tensor_attention_mask, visual_emb_mask], dim=1)

        # Tokenize answers if provided.
        if answers is not None:
            answers = [random.choice(answer) for answer in answers]
            labels = self.tokenizer(answers, return_tensors='pt', padding=True)
            labels.input_ids[labels.input_ids == self.tokenizer.pad_token_id] = -100
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
        # NEW: Get entity information (if available) from the batch.
        entity_tags = batch.get('entity_tags')
        entity_boxes = batch.get('entity_boxes')
        bs = len(question)

        if self.page_retrieval == 'logits':
            num_pages = batch['num_pages']
            outputs = []
            pred_answers = []
            pred_answer_pages = []
            pred_answers_conf = []

            for batch_idx in range(bs):
                # For logits-based page retrieval, replicate the question as needed.
                inp_embeds, att_mask, _ = self.prepare_inputs_for_vqa(
                    [question[batch_idx]] * num_pages[batch_idx],
                    words[batch_idx],
                    boxes[batch_idx],
                    images[batch_idx],
                    # Pass entity info if present.
                    entity_tags=entity_tags[batch_idx] if entity_tags is not None else None,
                    entity_boxes=entity_boxes[batch_idx] if entity_boxes is not None else None
                )
                pred_answer, logits = self.get_answer_from_model_output(inp_embeds, att_mask)
                max_logits = -999999
                answer_page = None
                best_answer = None
                for page_ix in range(len(inp_embeds)):
                    if logits[page_ix] > max_logits:
                        max_logits = logits[page_ix]
                        answer_page = page_ix
                        best_answer = pred_answer[page_ix]

                outputs.append(None)
                pred_answers.append(best_answer)
                pred_answer_pages.append(answer_page)
                pred_answers_conf.append(max_logits)

        else:
            inp_embeds, att_mask, labels = self.prepare_inputs_for_vqa(
                question, words, boxes, images, answers,
                entity_tags=entity_tags, entity_boxes=entity_boxes
            )
            outputs = self.model(inputs_embeds=inp_embeds, attention_mask=att_mask, labels=labels)
            if return_pred_answer:
                pred_answers, pred_answers_conf = self.get_answer_from_model_output(inp_embeds, att_mask)
            else:
                pred_answers, pred_answers_conf = None, None

            if self.page_retrieval == 'oracle':
                pred_answer_pages = batch['answer_page_idx']
            else:
                pred_answer_pages = None

        return outputs, pred_answers, pred_answer_pages, pred_answers_conf

    def get_answer_from_model_output(self, input_embeds, attention_mask):
        output = self.model.generate(inputs_embeds=input_embeds,
                                     attention_mask=attention_mask,
                                     output_scores=True,
                                     return_dict_in_generate=True,
                                     output_attentions=True)
        pred_answers = self.tokenizer.batch_decode(output['sequences'], skip_special_tokens=True)
        pred_answers_conf = model_utils.get_generative_confidence(output)
         # Debug prints to check model outputs.
        print("=== In get_answer_from_model_output ===")
        # print("Generated sequences shape:", output["sequences"].shape)
        # print("Sample generated sequence (token ids):", output["sequences"][0][:20])
        print("Decoded prediction sample:", pred_answers[0])
        print("Confidence sample:", pred_answers_conf[0] if pred_answers_conf else "None")

        return pred_answers, pred_answers_conf


