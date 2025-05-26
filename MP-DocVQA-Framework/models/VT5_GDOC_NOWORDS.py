# VT5_GDOC_NOWORDS - Corrected Version (Only Question, No OCR Tokens)

import random
import numpy as np

import torch
import torch.nn as nn
from transformers import T5Tokenizer, T5ForConditionalGeneration
import models._model_utils as model_utils
from models._modules import CustomT5Config, SpatialEmbeddings, VisualEmbeddings
from PIL import Image


class VT5_GDOC_NOWORDS:
    def __init__(self, config):
        self.batch_size = config['batch_size']
        self.tokenizer = T5Tokenizer.from_pretrained(config['model_weights'])
        self.model = T5ForConditionalGeneration.from_pretrained(config['model_weights'])

        self.page_retrieval = config['page_retrieval'].lower() if 'page_retrieval' in config else None
        self.max_source_length = config.get('max_source_length', 512)

        t5_config = CustomT5Config.from_pretrained(config['model_weights'])
        t5_config.visual_module_config = config['visual_module']

        self.spatial_embedding = SpatialEmbeddings(t5_config)
        self.visual_embedding = VisualEmbeddings(t5_config)

        # Move the embedding modules to the same device as the model
        device = config['device']
        self.spatial_embedding = self.spatial_embedding.to(device)
        self.visual_embedding = self.visual_embedding.to(device)
        self.model = self.model.to(device)

    def parallelize(self):
        self.model = nn.DataParallel(self.model)

    # EXACT SAME LOGIC AS VT5_GDOC, BUT NO OCR FOR LOOP

    def prepare_inputs_for_vqa(self, question, words, boxes, images, answers=None, graphdoc_embeds=None, graphdoc_masks=None):
        bs = len(question)
        prompt_text = ["question: {:s}  context: ".format(q) for q in question]

        # We'll store each sequence's IDs and bboxes
        longest_seq = 0
        batch_input_ids = []
        batch_input_boxes = []

        # Hard-coded bounding box for question tokens
        Q_BOX = [0, 0, 1000, 1000]
        EOS_BOX = [0, 0, 0, 0]

        for i in range(bs):
            # 1) Tokenize prompt
            tokenized_prompt = self.tokenizer(prompt_text[i])
            # e.g., tokenized_prompt.input_ids = [ <some tokens> , <eos>]
            # we drop last because original code does "[:-1]"
            question_ids = tokenized_prompt.input_ids[:-1]  # remove final eos
            question_boxes = [Q_BOX] * len(question_ids)     # repeated per token

            # 2) (NO OCR words, so skip the for loop that extends question_ids with words)

            # 3) Append final eos token + eos_box
            question_ids = question_ids[:self.max_source_length-1] + [self.tokenizer.eos_token_id]
            question_boxes = question_boxes[:self.max_source_length-1] + [EOS_BOX]

            batch_input_ids.append(question_ids)
            batch_input_boxes.append(np.array(question_boxes, dtype=np.int32))
            # track longest seq to pad later
            longest_seq = max(longest_seq, len(question_ids))

        # PADDING
        tensor_input_ids = torch.full([bs, longest_seq], fill_value=self.tokenizer.pad_token_id, dtype=torch.long)
        tensor_boxes = torch.full([bs, longest_seq, 4], fill_value=0, dtype=torch.long)
        tensor_attention_mask = torch.zeros([bs, longest_seq], dtype=torch.long)

        for i in range(bs):
            seq_len = len(batch_input_ids[i])
            tensor_input_ids[i, :seq_len] = torch.LongTensor(batch_input_ids[i])
            tensor_boxes[i, :seq_len] = torch.from_numpy(batch_input_boxes[i])
            tensor_attention_mask[i, :seq_len] = 1

        # Move to device
        tensor_input_ids = tensor_input_ids.to(self.model.device)
        tensor_boxes = tensor_boxes.to(self.model.device)
        tensor_attention_mask = tensor_attention_mask.to(self.model.device)

        # 4) Embeddings
        semantic_embedding = self.model.shared(tensor_input_ids)
        spatial_embedding = self.spatial_embedding(tensor_boxes)
        visual_embedding, visual_emb_mask = self.visual_embedding(images)

        input_embeds = semantic_embedding + spatial_embedding
        input_embeds = torch.cat([input_embeds, visual_embedding], dim=1)

        # 5) Attention mask
        tensor_attention_mask = torch.cat([tensor_attention_mask, visual_emb_mask], dim=1)

        # 6) Graphdoc if available
        if graphdoc_embeds is not None and graphdoc_masks is not None:
            graphdoc_embeds = graphdoc_embeds.to(self.model.device)
            graphdoc_masks = graphdoc_masks.to(self.model.device)

            print("INITIAL> input_embeds and attention mask", input_embeds.shape, tensor_attention_mask.shape)
            print("graphdoc_embeds and graphdoc_masks shapes", graphdoc_embeds.shape, graphdoc_masks.shape)

            input_embeds = torch.cat([input_embeds, graphdoc_embeds], dim=1)
            tensor_attention_mask = torch.cat([tensor_attention_mask, graphdoc_masks], dim=1)

            print("FINAL> input_embeds and attention mask", input_embeds.shape, tensor_attention_mask.shape)

        

        # 7) Tokenize answers
        if answers is not None:
            answers = [random.choice(ans) for ans in answers]
            labels = self.tokenizer(answers, return_tensors='pt', padding=True).input_ids
            labels[labels == self.tokenizer.pad_token_id] = -100
            labels = labels.to(self.model.device)
        else:
            labels = None

        return input_embeds, tensor_attention_mask, labels


    def forward(self, batch, return_pred_answer=False):
        question = batch['questions']
        images = batch['images']
        answers = batch['answers']

        graphdoc_embeds = batch.get('graphdoc_embeds')
        graphdoc_masks = batch.get('graphdoc_masks')

        bs = len(question)

        if self.page_retrieval == 'logits':
            num_pages = batch['num_pages']
            outputs = []
            pred_answers = []
            pred_answer_pages = []
            pred_answers_conf = []

            for batch_idx in range(bs):
                input_embeds, attention_mask, _ = self.prepare_inputs_for_vqa(
                    [question[batch_idx]] * num_pages[batch_idx], None, None, [images[batch_idx]] * num_pages[batch_idx]
                )  # Fixed incorrect words/boxes pass

                pred_answer, logits = self.get_answer_from_model_output(input_embeds, attention_mask)

                max_logits = -999999
                answer_page = None
                best_answer = None
                for page_ix in range(len(input_embeds)):
                    if logits[page_ix] > max_logits:
                        max_logits = logits[page_ix]
                        answer_page = page_ix
                        best_answer = pred_answer[page_ix]

                outputs.append(None)  # During inference, outputs are not used.
                pred_answers.append(best_answer)
                pred_answer_pages.append(answer_page)
                pred_answers_conf.append(max_logits)

        else:
            input_embeds, attention_mask, labels = self.prepare_inputs_for_vqa(
                question, None, None, images, answers,
                graphdoc_embeds=batch.get('graphdoc_embeds'),
                graphdoc_masks=batch.get('graphdoc_masks')
            )

            outputs = self.model(inputs_embeds=input_embeds, attention_mask=attention_mask, labels=labels)

            pred_answers, pred_answers_conf = self.get_answer_from_model_output(input_embeds, attention_mask) if return_pred_answer else None

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
