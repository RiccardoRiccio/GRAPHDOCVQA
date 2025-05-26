# VT5_GDOC_ModifiedDecoder.py

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    T5Config,
    T5PreTrainedModel,
)
from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers.models.t5.modeling_t5 import (
    T5LayerSelfAttention,
    T5LayerCrossAttention,
    T5LayerFF,
    T5LayerNorm,
)
from models._model_utils import get_generative_confidence  # Ensure this utility is correctly implemented
from models._modules import CustomT5Config, SpatialEmbeddings, VisualEmbeddings  # Ensure these modules are correctly implemented
from PIL import Image
import PIL

# ======================
# Custom Decoder Block
# ======================

class MultiModalT5Block(nn.Module):
    """
    A T5 decoder block augmented with an additional cross-attention layer for multi-modal embeddings.
    Includes a gating mechanism to balance textual and multi-modal information.
    """
    def __init__(self, config):
        super(MultiModalT5Block, self).__init__()
        self.is_decoder = config.is_decoder

        # Standard T5 decoder components
        self.self_attn = T5LayerSelfAttention(config, has_relative_attention_bias=False)
        if self.is_decoder:
            self.cross_attn = T5LayerCrossAttention(config, has_relative_attention_bias=False)
        self.ff = T5LayerFF(config)

        # Extra cross-attention for multi-modal embeddings
        self.multi_modal_cross_attn = T5LayerCrossAttention(config, has_relative_attention_bias=False)

        # Gating network to combine standard and multi-modal cross-attention
        self.gate_net = nn.Sequential(
            nn.Linear(config.d_model, 256),
            nn.ReLU(),
            nn.Linear(256, 2),
            nn.Softmax(dim=-1)  # Ensures the weights sum to 1
        )

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        multi_modal_embeds=None,
        multi_modal_mask=None,
        layer_head_mask=None,
        cross_attn_head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
    ):
        residual = hidden_states

        # ----- Self-Attention -----
        self_attn_outputs = self.self_attn(
            hidden_states=hidden_states,
            mask=attention_mask,
            layer_head_mask=layer_head_mask,
            past_key_value=None,  # Not handling past_key_value for simplicity
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        hidden_states = residual + self_attn_outputs[0]  # Add & Norm

        # ----- Cross-Attention to Encoder (Textual) -----
        if self.is_decoder and encoder_hidden_states is not None:
            residual = hidden_states
            cross_attn_outputs = self.cross_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                mask=encoder_attention_mask,
                layer_head_mask=cross_attn_head_mask,
                past_key_value=None,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            hidden_states = residual + cross_attn_outputs[0]  # Add & Norm

        # ----- Extra Cross-Attention to Multi-Modal Embeddings -----
        if multi_modal_embeds is not None and multi_modal_mask is not None:
            residual = hidden_states
            multi_modal_attn_outputs = self.multi_modal_cross_attn(
                hidden_states=hidden_states,
                key_value_states=multi_modal_embeds,
                mask=multi_modal_mask,
                layer_head_mask=None,  # Assuming no head mask for multi-modal
                past_key_value=None,
                use_cache=use_cache,
                output_attentions=False,
            )
            multi_modal_hidden = multi_modal_attn_outputs[0]

            # Compute gating weights
            summary_vector = multi_modal_hidden.mean(dim=1)  # [B, D]
            gating_weights = self.gate_net(summary_vector)    # [B, 2]

            vt5_weight = gating_weights[:, 0].unsqueeze(1).unsqueeze(2)       # [B, 1, 1]
            graphdoc_weight = gating_weights[:, 1].unsqueeze(1).unsqueeze(2)  # [B, 1, 1]

            # Fuse the multi-modal cross-attention with the standard hidden_states
            hidden_states = vt5_weight * multi_modal_hidden + graphdoc_weight * hidden_states  # [B, T, D]

        # ----- Feed-Forward Network -----
        residual = hidden_states
        ff_outputs = self.ff(hidden_states)
        hidden_states = residual + ff_outputs[0]  # Add & Norm

        return hidden_states, None  # No additional outputs

# ======================
# Custom Decoder Stack
# ======================

class MultiModalT5Decoder(nn.Module):
    """
    A T5 decoder stack that uses MultiModalT5Block for each decoder layer.
    """
    def __init__(self, config):
        super(MultiModalT5Decoder, self).__init__()
        self.block = nn.ModuleList([MultiModalT5Block(config) for _ in range(config.num_decoder_layers)])
        self.final_layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        decoder_input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        multi_modal_embeds=None,
        multi_modal_mask=None,
        **kwargs,
    ):
        # Embed decoder input ids
        if decoder_input_ids is not None:
            inputs_embeds = self.shared(decoder_input_ids)  # [B, T, D]
        else:
            raise ValueError("decoder_input_ids should be provided")

        hidden_states = inputs_embeds

        # Iterate through each block
        for block in self.block:
            hidden_states, _ = block(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                multi_modal_embeds=multi_modal_embeds,
                multi_modal_mask=multi_modal_mask,
                use_cache=False,  # Simplification: not handling caching
                output_attentions=False,
            )

        # Final layer norm and dropout
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states

# ======================
# Custom T5 Model
# ======================

class MultiModalT5ForConditionalGeneration(T5PreTrainedModel):
    """
    A T5 model with a modified decoder that can attend to multi-modal embeddings.
    """
    def __init__(self, config: T5Config):
        super(MultiModalT5ForConditionalGeneration, self).__init__(config)
        self.shared = nn.Embedding(config.vocab_size, config.d_model)
        self.encoder = T5ForConditionalGeneration(config).encoder  # Use standard T5 encoder
        self.decoder = MultiModalT5Decoder(config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_outputs=None,
        encoder_attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        multi_modal_embeds=None,
        multi_modal_mask=None,
        labels=None,
        use_cache=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=None,
    ):
        # Get encoder outputs
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
        hidden_states = encoder_outputs.last_hidden_state  # [B, S, D]

        # Prepare decoder inputs (shifted)
        if labels is not None and decoder_input_ids is None:
            decoder_input_ids = self._shift_right(labels)

        # Pass through decoder
        decoder_hidden_states = self.decoder(
            decoder_input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            multi_modal_embeds=multi_modal_embeds,
            multi_modal_mask=multi_modal_mask,
        )  # [B, T, D]

        # Compute logits
        logits = self.lm_head(decoder_hidden_states)  # [B, T, V]

        loss = None
        if labels is not None:
            # Compute loss
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        return Seq2SeqLMOutput(
            loss=loss,
            logits=logits,
            past_key_values=None,  # Not handling past_key_values for simplicity
            decoder_hidden_states=None,
            decoder_attentions=None,
            cross_attentions=None,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=None,
            encoder_attentions=None,
        )

# ======================
# VQA Class with Modified Decoder
# ======================

class VT5_GDOC_ModifiedDecoder:
    """
    VQA class using the MultiModalT5ForConditionalGeneration model with a modified decoder.
    """
    def __init__(self, config):
        """
        Initializes the VQA model with multi-modal capabilities.
        
        Args:
            config (dict): Configuration dictionary containing necessary parameters.
        """
        self.batch_size = config['batch_size']
        self.tokenizer = T5Tokenizer.from_pretrained(config['model_weights'])
        self.model = MultiModalT5ForConditionalGeneration.from_pretrained(config['model_weights'])

        self.page_retrieval = config.get('page_retrieval', None)
        self.max_source_length = config.get('max_source_length', 512)

        # Custom T5 Configuration
        t5_config = CustomT5Config.from_pretrained(config['model_weights'])
        t5_config.visual_module_config = config['visual_module']

        # Embedding Modules
        self.spatial_embedding = SpatialEmbeddings(t5_config)
        self.visual_embedding = VisualEmbeddings(t5_config)

        # Move the embedding modules to the same device as the model
        device = config['device']  # typically 'cuda'
        self.spatial_embedding = self.spatial_embedding.to(device)
        self.visual_embedding = self.visual_embedding.to(device)

        # Ensure the main model is also on the correct device
        self.model = self.model.to(device)

    def parallelize(self):
        """
        Enables DataParallel for the model.
        """
        self.model = nn.DataParallel(self.model)

    def prepare_inputs_for_vqa(self, question, words, boxes, images, answers=None, graphdoc_embeds=None, graphdoc_masks=None):
        """
        Prepares the inputs for the VQA task, including handling multi-modal embeddings.
        
        Args:
            question (List[str]): List of question strings.
            words (List[List[str]]): List of lists of context words.
            boxes (List[List[List[int]]]): List of lists of bounding boxes for each word.
            images (List[PIL.Image.Image] or List[torch.Tensor]): List of images or preprocessed tensors.
            answers (List[List[str]], optional): List of lists of possible answers.
            graphdoc_embeds (torch.Tensor, optional): Precomputed GraphDoc embeddings [B, S_gdoc, D].
            graphdoc_masks (torch.Tensor, optional): Attention masks for GraphDoc embeddings [B, S_gdoc].
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                - vt5_embedding: Fused textual and visual embeddings [B, S_total, D].
                - attention_mask: Attention mask for vt5_embedding [B, S_total].
                - multi_modal_embeds: GraphDoc embeddings [B, S_gdoc, D].
                - multi_modal_mask: Attention mask for GraphDoc embeddings [B, S_gdoc].
                - labels: Tokenized answer labels [B, L].
        """
        bs = len(words)
        prompt_text = [f"question: {q}  context: " for q in question]
        prompt_box = [0, 0, 1000, 1000]
        eos_box = [0, 0, 0, 0]
        padding_box_value = 0  # To become [0, 0, 0, 0] array.

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

        # Compute embeddings
        semantic_embedding = self.model.shared(tensor_input_ids)      # [B, S_vt5, D]
        spatial_embedding = self.spatial_embedding(tensor_boxes)      # [B, S_vt5, D]
        visual_embedding, visual_emb_mask = self.visual_embedding(images)  # [B, S_visual, D], [B, S_visual]

        # Fuse semantic and spatial embeddings
        vt5_embedding = semantic_embedding + spatial_embedding         # [B, S_vt5, D]
        vt5_embedding = torch.cat([vt5_embedding, visual_embedding], dim=1)  # [B, S_vt5 + S_visual, D]
        tensor_attention_mask = torch.cat([tensor_attention_mask, visual_emb_mask], dim=1)  # [B, S_total]

        # Handle GraphDoc embeddings if provided
        if graphdoc_embeds is not None and graphdoc_masks is not None:
            # Move to device
            graphdoc_embeds = graphdoc_embeds.to(vt5_embedding.device)
            graphdoc_masks = graphdoc_masks.to(tensor_attention_mask.device)

            # Assume graphdoc_embeds are [B, S_gdoc, D]
            multi_modal_embeds = graphdoc_embeds  # [B, S_gdoc, D]
            multi_modal_mask = graphdoc_masks      # [B, S_gdoc]
        else:
            multi_modal_embeds = None
            multi_modal_mask = None

        # Tokenize answers
        if answers is not None:
            # Randomly select one answer from the list of possible answers for each instance
            answers = [random.choice(a_list) for a_list in answers]
            labels = self.tokenizer(answers, return_tensors='pt', padding=True).input_ids  # [B, L]
            labels[labels == self.tokenizer.pad_token_id] = -100  # Ignore padding tokens in loss
            labels = labels.to(self.model.device)
        else:
            labels = None

        return vt5_embedding, tensor_attention_mask, multi_modal_embeds, multi_modal_mask, labels

    def forward(self, batch, return_pred_answer=False):
        """
        Forward pass for the VQA model.
        
        Args:
            batch (dict): Dictionary containing batch data.
            return_pred_answer (bool, optional): Whether to return predicted answers. Defaults to False.
        
        Returns:
            Tuple: (outputs, pred_answers, pred_answer_pages, pred_answers_conf)
        """
        # Unpack batch
        question = batch['questions']     # List of strings
        words = batch['words']            # List of lists of words
        boxes = batch['boxes']            # List of lists of bounding boxes
        images = batch['images']          # List of PIL Images or preprocessed tensors
        answers = batch.get('answers', None)        # List of lists of possible answers

        graphdoc_embeds = batch.get('graphdoc_embeds')  # [B, S_gdoc, D]
        graphdoc_masks = batch.get('graphdoc_masks')    # [B, S_gdoc]

        # Prepare inputs
        vt5_embedding, attention_mask, multi_modal_embeds, multi_modal_mask, labels = self.prepare_inputs_for_vqa(
            question, words, boxes, images, answers, graphdoc_embeds, graphdoc_masks
        )

        # Forward pass through the model
        outputs = self.model(
            input_ids=None,  # Not using input_ids directly
            attention_mask=None,  # Not used since encoder handles attention
            encoder_outputs=None,  # Let the model encode input_ids internally
            encoder_attention_mask=None,  # Let the model handle encoding
            decoder_input_ids=None,  # Let the model handle decoder inputs
            decoder_attention_mask=attention_mask,  # Pass the attention mask to the decoder
            multi_modal_embeds=multi_modal_embeds,   # [B, S_gdoc, D]
            multi_modal_mask=multi_modal_mask,       # [B, S_gdoc]
            labels=labels,                           # [B, L]
        )

        # Handle predictions if needed
        pred_answers, pred_answers_conf = None, None
        if return_pred_answer:
            # Implement generation logic with multi-modal embeddings
            # This requires a custom generate method, which is beyond this example
            pass

        # Handle page retrieval as per original code, if needed
        if self.page_retrieval == 'oracle':
            pred_answer_pages = batch['answer_page_idx']
        elif self.page_retrieval == 'concat':
            pred_answer_pages = None
        else:
            pred_answer_pages = None

        return outputs, pred_answers, pred_answer_pages, pred_answers_conf

    # Optionally, implement a custom generate method to handle multi-modal embeddings during inference
    # This is complex and may require modifying the generate method in the model

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