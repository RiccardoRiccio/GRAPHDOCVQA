# #######################
### APPLY CROSSATTENTION BETWEEN GRAPHDOC AND T5, THEN CONCATENATE TO ORIGINAL GDOC AND APPLY GATING
########################import torchimport torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

# Transformers & GraphDoc imports
from transformers import T5Tokenizer, T5ForConditionalGeneration
from layoutlmft.models.graphdoc.configuration_graphdoc import GraphDocConfig
from layoutlmft.models.graphdoc.modeling_graphdoc import GraphDocForEncode
from models._modules import CustomT5Config, SpatialEmbeddings  # spatial embedding module (same as GraphDoc)
import models._model_utils as model_utils  # e.g. get_generative_confidence

#############################################
# Helper Functions (copied from GraphDoc code)
#############################################
def scale_boxes(boxes, orig_width, orig_height, target_width=512, target_height=512):
    """
    Scales boxes (Tensor of shape [B, L, 4] in original image coordinates)
    to the target dimensions. The boxes are assumed to be in float.
    Returns integer (rounded) boxes.
    """
    ratio_w = target_width / orig_width
    ratio_h = target_height / orig_height
    # Multiply each coordinate by the appropriate ratio:
    boxes_scaled = boxes.float() * torch.tensor([ratio_w, ratio_h, ratio_w, ratio_h], device=boxes.device)
    return torch.round(boxes_scaled).long()

def merge2d(tensors, pad_id):
    dim1 = max([s.shape[0] for s in tensors])
    dim2 = max([s.shape[1] for s in tensors])
    out = tensors[0].new(len(tensors), dim1, dim2).fill_(pad_id)
    for i, s in enumerate(tensors):
        out[i, :s.shape[0], :s.shape[1]] = s
    return out

def merge3d(tensors, pad_id):
    dim1 = max([s.shape[0] for s in tensors])
    dim2 = max([s.shape[1] for s in tensors])
    dim3 = max([s.shape[2] for s in tensors])
    out = tensors[0].new(len(tensors), dim1, dim2, dim3).fill_(pad_id)
    for i, s in enumerate(tensors):
        out[i, :s.shape[0], :s.shape[1], :s.shape[2]] = s
    return out

def mask1d(tensors, pad_id):
    lengths = [len(s) for s in tensors]
    out = tensors[0].new(len(tensors), max(lengths)).fill_(pad_id)
    for i, s in enumerate(tensors):
        out[i, :len(s)] = 1
    return out

#############################################
# Gated Cross-Attention Fusion Module
#############################################
class GraphDocCrossAttentionWithGate(nn.Module):
    def __init__(self, hidden_size=768, num_heads=12):
        super(GraphDocCrossAttentionWithGate, self).__init__()
        # Here we use a multihead attention module.
        # The idea is: each GraphDoc (entity) embedding (query) attends over the T5 token embeddings (key/value).
        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True)
        # The gate is computed from the concatenation of the original and cross-attended embeddings.
        self.gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Sigmoid()
        )
        # Optional final projection.
        self.proj = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, original, t5_tokens, key_padding_mask=None):
        # original: [B, L_entity, H]
        # t5_tokens: [B, L_word, H]
        cross_out, _ = self.cross_attn(query=original, key=t5_tokens, value=t5_tokens, key_padding_mask=key_padding_mask)
        concat_out = torch.cat([original, cross_out], dim=-1)  # [B, L_entity, 2H]
        gate_val = self.gate(concat_out)  # [B, L_entity, H] values in [0,1]
        fused = gate_val * original + (1 - gate_val) * cross_out
        fused = self.proj(fused)
        return fused

#############################################
# Document VQA Model
#############################################
class DocumentVQA(nn.Module):
    def __init__(self, config):
        """
        Implements a Document VQA model with two encoder branches and a T5 decoder.
        
        - Frozen T5 encoder branch (word-level): Computes word embeddings as:
            t5_proj( T5_shared(token_ids) ) + SpatialEmbedding(token_box)
          (Here the spatial embedding is computed using the same SpatialEmbeddings module as GraphDoc.)
          
        - Trainable GraphDoc encoder branch (entity-level): Processes OCR entity texts and boxes.
        
        - A gated cross-attention module fuses the two streams: each entity embedding (GraphDoc) attends over T5 word embeddings,
          and a gate fuses the original GraphDoc and the cross-attended result.
        
        - The enriched GraphDoc embeddings are used as encoder_hidden_states for the T5 decoder.
        
        config: dictionary with keys:
          - "t5_weights": identifier for T5 weights (e.g. "t5-base")
          - "graphdoc_weights": path or identifier for pretrained GraphDoc weights
          - "device": "cuda" or "cpu"
        """
        super(DocumentVQA, self).__init__()
        device = config['device']

        # ---------- Frozen T5 Encoder Branch (Word-Level) ----------
        self.t5_tokenizer = T5Tokenizer.from_pretrained(config['t5_weights'])
        self.t5_model = T5ForConditionalGeneration.from_pretrained(config['t5_weights'])
        # Freeze T5 encoder parameters.
        for param in self.t5_model.encoder.parameters():
            param.requires_grad = False
        # Projection layer for T5 embeddings.
        self.t5_proj = nn.Linear(768, 768)
        t5_config = CustomT5Config.from_pretrained(config['t5_weights'])
        self.spatial_embedding = SpatialEmbeddings(t5_config).to(device)
        
        # ---------- Trainable GraphDoc Encoder Branch (Entity-Level) ----------
        graphdoc_config = GraphDocConfig.from_pretrained(config['graphdoc_weights'])
        self.graphdoc_encoder = GraphDocForEncode.from_pretrained(config['graphdoc_weights'], config=graphdoc_config)
        self.graphdoc_encoder = self.graphdoc_encoder.to(device)
        
        # ---------- Gated Cross-Attention Fusion Module ----------
        self.gated_cross_attn = GraphDocCrossAttentionWithGate(hidden_size=768, num_heads=12)
        
        # ---------- T5 Decoder ----------
        self.t5_model = self.t5_model.to(device)
        self.t5_proj = self.t5_proj.to(device)
    
    def prepare_t5_branch(self, question, words, boxes, images):
        """
        Prepares the T5 branch.
        
        For each sample, a prompt is built of the form:
            "question: <q>  context: "
        Then OCR word tokens are appended.
        
        Each token embedding is computed as:
            t5_proj( T5_shared(token) ) + SpatialEmbedding(token_box)
            
        The provided word-level boxes (which are in the original image scale)
        are scaled to 512×512 using the first image's dimensions.
        """
        bs = len(words)
        prompt_text = ["question: {:s}  context: ".format(q) for q in question]
        prompt_box = [0, 0, 512, 512]
        eos_box = [0, 0, 0, 0]
        padding_box_value = 0

        longest_seq = 0
        batch_input_ids = []
        batch_input_boxes = []
        for i in range(bs):
            tokenized_prompt = self.t5_tokenizer(prompt_text[i])
            input_ids = tokenized_prompt.input_ids[:-1]
            input_boxes = [prompt_box] * len(input_ids)
            for word, box in zip(words[i], boxes[i]):
                tokenized_word = self.t5_tokenizer(word).input_ids[:-1]
                input_ids.extend(tokenized_word)
                input_boxes.extend([box] * len(tokenized_word))
            # Truncate to 511 tokens and append EOS.
            batch_input_ids.append(input_ids[:511] + [self.t5_tokenizer.eos_token_id])
            batch_input_boxes.append(np.concatenate([input_boxes[:511], np.array([eos_box])]))
            longest_seq = min(max(longest_seq, len(input_ids) + 1), 512)
        
        tensor_input_ids = torch.full((bs, longest_seq), fill_value=self.t5_tokenizer.pad_token_id, dtype=torch.long)
        tensor_boxes = torch.full((bs, longest_seq, 4), fill_value=padding_box_value, dtype=torch.long)
        tensor_attention_mask = torch.zeros((bs, longest_seq), dtype=torch.long)
        
        for i in range(bs):
            seq_len = len(batch_input_ids[i])
            tensor_input_ids[i, :seq_len] = torch.tensor(batch_input_ids[i], dtype=torch.long)
            tensor_boxes[i, :seq_len] = torch.from_numpy(batch_input_boxes[i])
            tensor_attention_mask[i, :seq_len] = 1
        
        tensor_input_ids = tensor_input_ids.to(self.t5_model.device)
        tensor_boxes = tensor_boxes.to(self.t5_model.device)
        tensor_attention_mask = tensor_attention_mask.to(self.t5_model.device)
        
        # Scale the word-level boxes.
        image0 = images[0]
        orig_width, orig_height = image0.size  # PIL: (width, height)
        tensor_boxes = scale_boxes(tensor_boxes.float(), orig_width, orig_height, target_width=512, target_height=512)
        
        semantic_embeds = self.t5_model.shared(tensor_input_ids)  # [B, L_word, 768]
        projected_embeds = self.t5_proj(semantic_embeds)            # [B, L_word, 768]
        spatial_embeds = self.spatial_embedding(tensor_boxes)       # [B, L_word, 768]
        t5_branch = projected_embeds + spatial_embeds              # [B, L_word, 768]
        return t5_branch, tensor_attention_mask

    def forward(self, batch):
        """
        Expects a batch dictionary with keys:
          - "questions": [B] list of question strings.
          - "words": list (B) of lists of OCR word strings.
          - "boxes": [B, L_word, 4] word-level boxes (original scale).
          - "lines": list (B) of lists of entity strings.
          - "line_boxes": [B, L_entity, 4] entity-level boxes (original scale).
          - "images": either a single PIL image or a list of images.
          - "decoder_input_ids": [B, L_dec] target tokens for the T5 decoder.
          - "decoder_attention_mask": [B, L_dec] (optional) for the decoder.
        """
        device = self.t5_model.device

        # ----- T5 Encoder Branch (Word-Level) -----
        t5_branch, t5_attn_mask = self.prepare_t5_branch(
            question=batch['questions'],
            words=batch['words'],
            boxes=batch['boxes'],
            images=[batch['images']] if isinstance(batch['images'], Image.Image) else batch['images']
        )
        # t5_branch: [B, L_word, 768]

        # ----- GraphDoc Encoder Branch (Entity-Level) -----
        # Get entity texts and boxes (from the dataloader).
        graphdoc_texts = batch['lines']  # list of lists of entity strings
        graphdoc_boxes = torch.tensor(batch['line_boxes']).to(device)  # [B, L_entity, 4]
        if isinstance(batch['images'], Image.Image):
            image0 = batch['images']
        else:
            image0 = batch['images'][0]
        orig_width, orig_height = image0.size
        scaled_graphdoc_boxes = scale_boxes(graphdoc_boxes.float(), orig_width, orig_height, target_width=512, target_height=512)
        B, L_entity, _ = scaled_graphdoc_boxes.size()
        graphdoc_attn_mask = torch.ones(B, L_entity, device=device, dtype=torch.long)
        graphdoc_out = self.graphdoc_encoder(
            input_sentences=graphdoc_texts,
            bbox=scaled_graphdoc_boxes,
            attention_mask=graphdoc_attn_mask
        )
        # Take the first element of the GraphDoc encoder output as the embeddings.
        graphdoc_branch = graphdoc_out[0]  # [B, L_entity, 768]

        # ----- Cross-Attention with Gating -----
        # Let GraphDoc entity embeddings (queries) attend over T5 word embeddings.
        enriched_graphdoc = self.gated_cross_attn(original=graphdoc_branch, t5_tokens=t5_branch)
        # enriched_graphdoc: [B, L_entity, 768]

        # ----- T5 Decoder -----
        decoder_input_ids = batch['decoder_input_ids']  # [B, L_dec]
        decoder_attention_mask = batch.get('decoder_attention_mask', torch.ones_like(decoder_input_ids))
        # Use the enriched GraphDoc embeddings as the encoder_hidden_states.
        decoder_outputs = self.t5_model.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=enriched_graphdoc,
            encoder_attention_mask=None
        )
        sequence_output = decoder_outputs.last_hidden_state  # [B, L_dec, 768]
        logits = self.t5_model.lm_head(sequence_output)         # [B, L_dec, vocab_size]
        return logits

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

    # For T5 branch: simulate word-level boxes (original scale, e.g. for an image 1024x1024)
    t5_boxes = torch.randint(0, 1024, (batch_size, L_word, 4)).to(config["device"])
    # For entity branch: simulate entity-level boxes (original scale)
    graphdoc_boxes = torch.randint(0, 1024, (batch_size, L_entity, 4)).to(config["device"])
    # Dummy texts:
    t5_words = [["dummy"] * L_word for _ in range(batch_size)]
    graphdoc_texts = [["Entity text {}".format(i) for i in range(L_entity)] for _ in range(batch_size)]

    # Decoder inputs:
    decoder_input_ids = torch.randint(0, vocab_size, (batch_size, L_dec)).to(config["device"])
    decoder_attention_mask = torch.ones(batch_size, L_dec).to(config["device"])

    # Create a dummy PIL image with original size 1024x1024
    from PIL import Image
    dummy_img = Image.new("RGB", (1024, 1024), color="white")

    # Build a dummy batch dictionary (simulate dataloader output)
    batch = {
        "questions": ["What is the main font used?"] * batch_size,
        "words": t5_words,
        "boxes": t5_boxes,           # word-level boxes (original scale)
        "lines": graphdoc_texts,       # entity texts
        "line_boxes": graphdoc_boxes,  # entity-level boxes (original scale)
        "images": dummy_img,           # single PIL image (or list of images)
        "decoder_input_ids": decoder_input_ids,
        "decoder_attention_mask": decoder_attention_mask
    }

    # Forward pass
    logits = model(batch)
    print("Logits shape:", logits.shape)

    # To generate an answer using the T5 branch prompt:
    t5_branch, t5_mask = model.prepare_t5_branch(
        question=["What is the main font used?"] * batch_size,
        words=t5_words,
        boxes=t5_boxes.cpu().numpy(),
        images=[dummy_img]
    )
    pred_answers, pred_answers_conf = model.get_answer_from_model_output(t5_branch, t5_mask)
