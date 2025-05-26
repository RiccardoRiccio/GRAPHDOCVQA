import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

from transformers import T5Tokenizer, T5ForConditionalGeneration
from layoutlmft.models.graphdoc.configuration_graphdoc import GraphDocConfig
from layoutlmft.models.graphdoc.modeling_graphdoc import GraphDocForEncode
from models._modules import CustomT5Config, SpatialEmbeddings  # Your project’s spatial embedding module
import models._model_utils as model_utils  # For get_generative_confidence, etc.

#############################################
# Helper Functions
#############################################
def scale_boxes(boxes, orig_width, orig_height, target_width=512, target_height=512):
    """
    boxes: Tensor of shape [B, L, 4] in original pixel coordinates.
    Scales them to the target dimensions.
    """
    ratio_w = target_width / orig_width
    ratio_h = target_height / orig_height
    # Multiply each box coordinate by the respective ratio and round to int.
    boxes_scaled = boxes.float() * torch.tensor([ratio_w, ratio_h, ratio_w, ratio_h], device=boxes.device)
    return torch.round(boxes_scaled).long()

#############################################
# Gated Cross-Attention Fusion Module
#############################################
class GatedCrossAttentionFusion(nn.Module):
    def __init__(self, hidden_size=768, num_heads=12):
        super(GatedCrossAttentionFusion, self).__init__()
        # Cross-attention modules for each branch.
        self.cross_attn_t5 = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True)
        self.cross_attn_graphdoc = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True)
        # Gating layers: compute a gate value from concatenated features.
        self.gate_t5 = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Sigmoid()
        )
        self.gate_doc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Sigmoid()
        )
        # Final projection after concatenation.
        self.proj = nn.Linear(hidden_size * 2, hidden_size)
    
    def forward(self, t5_feats, graphdoc_feats, t5_attn_mask=None, graphdoc_attn_mask=None):
        # t5_feats: [B, L_word, H]
        # graphdoc_feats: [B, L_entity, H]
        cross_out_t5, _ = self.cross_attn_t5(query=t5_feats, key=graphdoc_feats, value=graphdoc_feats,
                                              key_padding_mask=graphdoc_attn_mask)
        cross_out_doc, _ = self.cross_attn_graphdoc(query=graphdoc_feats, key=t5_feats, value=t5_feats,
                                                    key_padding_mask=t5_attn_mask)
        # For T5 branch: gate between original and cross-attended output.
        fused_t5 = torch.cat([t5_feats, cross_out_t5], dim=-1)  # [B, L_word, 2H]
        gate_val_t5 = self.gate_t5(fused_t5)                    # [B, L_word, H]
        t5_fused = gate_val_t5 * t5_feats + (1 - gate_val_t5) * cross_out_t5
        
        # For GraphDoc branch:
        fused_doc = torch.cat([graphdoc_feats, cross_out_doc], dim=-1)  # [B, L_entity, 2H]
        gate_val_doc = self.gate_doc(fused_doc)                         # [B, L_entity, H]
        doc_fused = gate_val_doc * graphdoc_feats + (1 - gate_val_doc) * cross_out_doc
        
        # Concatenate the two sequences along the token dimension.
        fused_seq = torch.cat([t5_fused, doc_fused], dim=1)  # [B, L_word + L_entity, H]
        # Optionally, apply a projection.
        fused_seq = self.proj(fused_seq)  # [B, L_word + L_entity, H]
        return fused_seq

#############################################
# Document VQA Model
#############################################
class DocumentVQA(nn.Module):
    def __init__(self, config):
        """
        config: dictionary with keys:
          - 't5_weights': e.g. "t5-base"
          - 'graphdoc_weights': path or identifier for pretrained GraphDoc weights
          - 'device': "cuda" or "cpu"
        """
        super(DocumentVQA, self).__init__()
        device = config['device']

        # ---------- Frozen T5 Encoder Branch (Word-Level) ----------
        self.t5_tokenizer = T5Tokenizer.from_pretrained(config['t5_weights'])
        self.t5_model = T5ForConditionalGeneration.from_pretrained(config['t5_weights'])
        # Freeze T5 encoder parameters.
        for param in self.t5_model.encoder.parameters():
            param.requires_grad = False
        # A projection layer applied to T5 shared embeddings.
        self.t5_proj = nn.Linear(768, 768)
        # Use the same spatial embedding module as GraphDoc.
        t5_config = CustomT5Config.from_pretrained(config['t5_weights'])
        self.spatial_embedding = SpatialEmbeddings(t5_config)
        self.spatial_embedding = self.spatial_embedding.to(device)

        # ---------- Trainable GraphDoc Encoder Branch (Entity-Level) ----------
        graphdoc_config = GraphDocConfig.from_pretrained(config['graphdoc_weights'])
        self.graphdoc_encoder = GraphDocForEncode.from_pretrained(config['graphdoc_weights'], config=graphdoc_config)
        self.graphdoc_encoder = self.graphdoc_encoder.to(device)

        # ---------- Fusion Module ----------
        self.fusion = GatedCrossAttentionFusion(hidden_size=768, num_heads=12)

        # ---------- T5 Decoder ----------
        self.t5_model = self.t5_model.to(device)
        self.t5_proj = self.t5_proj.to(device)
    
    def prepare_inputs_for_vqa(self, question, words, boxes, images, answers=None):
        """
        This function builds a prompt for the T5 branch.
        It creates a prompt of the form: "question: <q>  context: " and then appends the OCR words.
        The prompt_box is set to [0, 0, 512, 512] since the target size is 512×512.
        It also scales the provided word-level boxes (which are in the original image scale)
        using the first image's dimensions.
        """
        bs = len(words)
        prompt_text = ["question: {:s}  context: ".format(q) for q in question]
        prompt_box = [0, 0, 512, 512]
        eos_box = [0, 0, 0, 0]
        padding_box_value = 0

        longest_seq = 0
        batch_input_ids = []
        batch_input_boxes = []
        for batch_idx in range(bs):
            tokenized_prompt = self.t5_tokenizer(prompt_text[batch_idx])
            input_ids = tokenized_prompt.input_ids[:-1]
            input_boxes = [prompt_box] * len(input_ids)
            for word, box in zip(words[batch_idx], boxes[batch_idx]):
                tokenized_word = self.t5_tokenizer(word).input_ids[:-1]
                input_ids.extend(tokenized_word)
                input_boxes.extend([box] * len(tokenized_word))
            batch_input_ids.append(input_ids[:511] + [self.t5_tokenizer.eos_token_id])
            batch_input_boxes.append(np.concatenate([input_boxes[:511], np.array([eos_box])]))
            longest_seq = min(max(longest_seq, len(input_ids) + 1), 512)

        tensor_input_ids = torch.full([bs, longest_seq], fill_value=self.t5_tokenizer.pad_token_id, dtype=torch.long)
        tensor_boxes = torch.full([bs, longest_seq, 4], fill_value=padding_box_value, dtype=torch.long)
        tensor_attention_mask = torch.zeros([bs, longest_seq], dtype=torch.long)

        for batch_idx in range(bs):
            seq_len = len(batch_input_ids[batch_idx])
            tensor_input_ids[batch_idx, :seq_len] = torch.LongTensor(batch_input_ids[batch_idx])
            tensor_boxes[batch_idx, :seq_len] = torch.from_numpy(batch_input_boxes[batch_idx])
            tensor_attention_mask[batch_idx, :seq_len] = 1

        tensor_input_ids = tensor_input_ids.to(self.t5_model.device)
        tensor_boxes = tensor_boxes.to(self.t5_model.device)
        tensor_attention_mask = tensor_attention_mask.to(self.t5_model.device)

        # Scale the boxes using the first image's dimensions.
        image = images[0]
        orig_width, orig_height = image.size  # PIL returns (width, height)
        tensor_boxes = scale_boxes(tensor_boxes.float(), orig_width, orig_height, target_width=512, target_height=512)
        semantic_embedding = self.t5_model.shared(tensor_input_ids)
        spatial_embedding = self.spatial_embedding(tensor_boxes)
        input_embeds = semantic_embedding + spatial_embedding
        return input_embeds, tensor_attention_mask

    def get_answer_from_model_output(self, input_embeds, attention_mask):
        output = self.t5_model.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            output_scores=True,
            return_dict_in_generate=True,
            output_attentions=True
        )
        pred_answers = self.t5_tokenizer.batch_decode(output['sequences'], skip_special_tokens=True)
        pred_answers_conf = model_utils.get_generative_confidence(output)
        print("=== In get_answer_from_model_output ===")
        print("Decoded prediction sample:", pred_answers[0])
        print("Confidence sample:", pred_answers_conf[0] if pred_answers_conf else "None")
        return pred_answers, pred_answers_conf

    def forward(self, batch):
        """
        Expects a batch (dictionary) from the dataloader with keys:
          - "questions": [B] string questions.
          - "words": list (B) of lists of OCR word strings.
          - "boxes": [B, L_word, 4] word-level boxes (original scale).
          - "lines": list (B) of lists of entity strings.
          - "line_boxes": [B, L_entity, 4] entity-level boxes (original scale).
          - "images": a list (or PIL image) for each sample.
          - (Optionally) "answers": list of answers.
        Additionally, the training code must provide decoder_input_ids
        for the T5 decoder (shifted target tokens) and appropriate attention masks.
        """
        device = batch['boxes'][0].device if isinstance(batch['boxes'], torch.Tensor) else torch.tensor(batch['boxes'][0]).device
        # Prepare T5 branch inputs using the prepare_inputs_for_vqa function.
        # Here we pass the question, words, and boxes (word-level) along with the image.
        input_embeds, t5_mask = self.prepare_inputs_for_vqa(
            question=batch['questions'],
            words=batch['words'],
            boxes=batch['boxes'],
            images=[batch['images']] if isinstance(batch['images'], Image.Image) else batch['images']
        )

        # For the GraphDoc branch, convert entity-level boxes and texts:
        graphdoc_texts = batch['lines']   # list of lists of entity texts
        # Assume line_boxes is provided as a NumPy array; convert to tensor.
        graphdoc_boxes = torch.tensor(batch['line_boxes']).to(self.t5_model.device)  # shape: [B, L_entity, 4]
        # Scale entity boxes using the same image dimensions:
        if isinstance(batch['images'], Image.Image):
            image = batch['images']
        else:
            image = batch['images'][0]
        orig_width, orig_height = image.size
        scaled_graphdoc_boxes = scale_boxes(graphdoc_boxes.float(), orig_width, orig_height, target_width=512, target_height=512)

        B, L_entity, _ = scaled_graphdoc_boxes.size()
        graphdoc_attention_mask = torch.ones(B, L_entity, device=self.t5_model.device, dtype=torch.long)
        graphdoc_out = self.graphdoc_encoder(
            input_sentences=graphdoc_texts,
            bbox=scaled_graphdoc_boxes,
            attention_mask=graphdoc_attention_mask
        )
        graphdoc_branch = graphdoc_out[0]  # [B, L_entity, 768]

        # ----- T5 Encoder Branch -----
        # We already have t5_branch as input_embeds (which is from T5 shared + spatial)
        # Apply the projection layer:
        t5_emb = self.t5_model.shared(batch['t5_input_ids']) if 't5_input_ids' in batch else None
        # (If you want to override the prompt-based embeddings, you can compute t5_emb separately.)
        # Here we use input_embeds from prepare_inputs_for_vqa as our T5 branch.
        # For this example, we assume input_embeds is [B, L_word, 768].
        # If not, adjust accordingly.

        # Optionally, if you have a pre-computed t5_input_ids in the batch, you can use them.
        # In our example, we simply use input_embeds as t5_branch.
        t5_branch = input_embeds  # [B, L_word, 768]

        # ----- Fusion -----
        fused_encoder_outputs = self.fusion(t5_branch, graphdoc_branch)
        # fused_encoder_outputs shape: [B, L_word + L_entity, 768]

        # For training, the batch should include decoder_input_ids and decoder_attention_mask.
        decoder_input_ids = batch['decoder_input_ids']  # [B, L_dec]
        decoder_attention_mask = batch.get('decoder_attention_mask', torch.ones_like(decoder_input_ids))
        # ----- T5 Decoder -----
        decoder_outputs = self.t5_model.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=fused_encoder_outputs,
            encoder_attention_mask=None
        )
        sequence_output = decoder_outputs.last_hidden_state  # [B, L_dec, 768]
        logits = self.t5_model.lm_head(sequence_output)         # [B, L_dec, vocab_size]
        return logits

#############################################
# Example Usage
#############################################
if __name__ == "__main__":
    # Example configuration:
    config = {
        "t5_weights": "t5-base",
        "graphdoc_weights": "pretrained_model/graphdoc",
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }
    model = DocumentVQA(config).to(config["device"])

    # Dummy inputs for testing:
    batch_size = 2
    L_word = 300   # Number of word-level tokens for T5 branch
    L_entity = 50  # Number of entity tokens for GraphDoc branch
    L_dec = 20     # Decoder sequence length
    vocab_size = model.t5_model.config.vocab_size

    # For T5 branch, create dummy token IDs (if not using prepare_inputs_for_vqa)
    t5_input_ids = torch.randint(0, vocab_size, (batch_size, L_word)).to(config["device"])
    # Simulate word boxes in original scale (assume original image size is 1024x1024)
    t5_boxes = torch.randint(0, 1024, (batch_size, L_word, 4)).to(config["device"])
    # For entity branch, create dummy entity texts and boxes
    graphdoc_texts = [["Entity text {}".format(i) for i in range(L_entity)] for _ in range(batch_size)]
    graphdoc_boxes = torch.randint(0, 1024, (batch_size, L_entity, 4)).to(config["device"])

    # Decoder inputs (shifted target tokens):
    decoder_input_ids = torch.randint(0, vocab_size, (batch_size, L_dec)).to(config["device"])
    t5_attention_mask = torch.ones(batch_size, L_word).to(config["device"])
    decoder_attention_mask = torch.ones(batch_size, L_dec).to(config["device"])

    # Create a dummy PIL image with original size 1024x1024
    from PIL import Image
    dummy_img = Image.new("RGB", (1024, 1024), color="white")

    # Build a dummy batch dictionary (as produced by your dataloader)
    batch = {
        "questions": ["What is the main font used?"] * batch_size,
        "words": [["dummy"] * L_word for _ in range(batch_size)],
        "boxes": t5_boxes,  # word-level boxes (tensor)
        "lines": [["Entity text {}".format(i) for i in range(L_entity)] for _ in range(batch_size)],
        "line_boxes": graphdoc_boxes,  # entity-level boxes (tensor)
        "images": dummy_img,  # single PIL image per sample
        # For decoder, you would normally tokenize the target answer:
        "decoder_input_ids": decoder_input_ids,
        "t5_input_ids": t5_input_ids,  # optional if available
        "decoder_attention_mask": decoder_attention_mask
    }

    logits = model(batch)
    print("Logits shape:", logits.shape)

    # To generate an answer using the T5 branch prompt:
    input_embeds, t5_mask = model.prepare_inputs_for_vqa(
        question=["What is the main font used?"] * batch_size,
        words=[["dummy"] * L_word for _ in range(batch_size)],
        boxes=t5_boxes.cpu().numpy(),
        images=[dummy_img]
    )
    pred_answers, pred_answers_conf = model.get_answer_from_model_output(input_embeds, t5_mask)
