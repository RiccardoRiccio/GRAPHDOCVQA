import torch
import torch.nn as nn
import cv2
import numpy as np
import models._model_utils as model_utils

import sys
sys.path.append("/home/rriccio/GraphDoc_VQA")

from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoModel, AutoTokenizer
from transformers.modeling_outputs import BaseModelOutput
from layoutlmft.models.graphdoc.configuration_graphdoc import GraphDocConfig
from layoutlmft.models.graphdoc.modeling_graphdoc import GraphDocForEncode


def scale_boxes(boxes, orig_w, orig_h, tgt_w=512, tgt_h=512):
    """Scale [B, L, 4] boxes from (orig_w,orig_h)→(tgt_w,tgt_h)."""
    ratios = torch.tensor([tgt_w/orig_w, tgt_h/orig_h, tgt_w/orig_w, tgt_h/orig_h],
                          device=boxes.device)
    return torch.round(boxes.float() * ratios).long()

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

def mask1d(tensors, pad_id=0):
    lengths = [len(s) for s in tensors]
    max_len = max(lengths)
    out = torch.zeros(len(tensors), max_len, dtype=torch.long, device=tensors[0].device)
    for i, length in enumerate(lengths):
        out[i, :length] = 1
    return out

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def extract_sentence_embeddings(contents, tokenizer, sentence_bert):
    encoded_input = tokenizer(contents, padding=True, truncation=True, return_tensors='pt')
    encoded_input = encoded_input.to(sentence_bert.device)
    with torch.no_grad():
        model_output = sentence_bert(**encoded_input)
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask']).cpu().numpy()
    return sentence_embeddings

class GDOCVQA_ONLYGLOBAL(nn.Module):
    """
    Corrected VQA using GraphDoc + T5 decoder:
     - Properly processes images and extracts sentence embeddings
     - Uses correct GraphDoc API with image, inputs_embeds, bbox, attention_mask
     - Question embedded as global node
    """
    def __init__(self,
                 graphdoc_ckpt: str,
                 sentence_bert_path: str,
                 t5_name: str="t5-base",
                 device: str="cuda"):
        super().__init__()
        self.device = device
        self.input_H = 512
        self.input_W = 512

        # 1) Load T5 for answer generation
        self.tokenizer = T5Tokenizer.from_pretrained(t5_name)
        self.t5 = T5ForConditionalGeneration.from_pretrained(t5_name).to(device)
        for p in self.t5.encoder.parameters():
            p.requires_grad = False

        # 2) Load GraphDoc encoder
        gd_cfg = GraphDocConfig.from_pretrained(graphdoc_ckpt)
        self.graphdoc = GraphDocForEncode.from_pretrained(graphdoc_ckpt,
                                                           config=gd_cfg).to(device)

        # 3) Load sentence BERT for text embeddings
        self.sent_tokenizer = AutoTokenizer.from_pretrained(sentence_bert_path)
        self.sentence_bert = AutoModel.from_pretrained(sentence_bert_path).to(device)
        self.sentence_bert.eval()
        for p in self.sentence_bert.parameters():
            p.requires_grad = False

        # 4) Projection layer
        self.proj_to_t5 = nn.Linear(gd_cfg.hidden_size,
                                    self.t5.config.d_model).to(device)

    def process_images(self, images):
        """Process PIL images to GraphDoc format"""
        processed_images = []
        
        for img in images:
            # Convert PIL to numpy array
            img_np = np.array(img)
            H, W = img_np.shape[:2]
            
            # Resize to 512x512
            img_resized = cv2.resize(img_np, dsize=(self.input_W, self.input_H))
            
            # Convert to tensor format (C, H, W)
            img_tensor = torch.from_numpy(img_resized.transpose(2, 0, 1).astype(np.float32))
            processed_images.append(img_tensor)
        
        # Merge into batch
        return merge3d(processed_images, 0).to(self.device)

    def get_answer_from_model_output(self, encoder_hidden_states, encoder_attention_mask):
        try:
            # Create BaseModelOutput from your encoder's last_hidden_state
            enc_out = BaseModelOutput(last_hidden_state=encoder_hidden_states)
            
            # Generate with carefully chosen parameters
            outputs = self.t5.generate(
                encoder_outputs=enc_out,
                attention_mask=encoder_attention_mask,
                # Generation length control
                max_length=32,        # Maximum answer length
                min_length=1,         # Ensure non-empty answers
                length_penalty=1.0,   # Balance between short/long answers
                # Beam search parameters
                num_beams=4,          # Number of beams for better exploration
                early_stopping=True,  # Stop when all beams finished
                # Output configuration for confidence scoring
                output_scores=True,           # Get token probabilities
                return_dict_in_generate=True  # Get structured output
            )
            
            pred_answers = self.tokenizer.batch_decode(
                outputs.sequences, 
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            pred_answers_conf = model_utils.get_generative_confidence(outputs)
            
            # Debug generation
            print("\nGeneration Debug:")
            print(f"Sequence shape: {outputs.sequences.shape}")
            print(f"Sample raw output: {outputs.sequences[0]}")
            print(f"Sample decoded: {pred_answers[0]}")
            
            return pred_answers, pred_answers_conf
            
        except Exception as e:
            print(f"Error in answer generation: {str(e)}")
            batch_size = encoder_hidden_states.size(0)
            return ["unknown"] * batch_size, [0.0] * batch_size

    def forward(self, batch, return_pred_answer=False):
        """
        batch must contain:
          - 'questions': list[str] of length B  # Note: changed from 'question' to 'questions'
          - 'lines': list[list[str]] of length B, each inner list = entity texts
          - 'line_boxes': list or torch.LongTensor [B, L_ent, 4] in orig image coords
          - 'images': list[PIL.Image] of length B
          - 'image_width': list[int] - original image widths
          - 'image_height': list[int] - original image heights
          - 'decoder_input_ids': LongTensor [B, L_dec] (for training)
          - 'decoder_attention_mask': LongTensor [B, L_dec] (for training)
        """
        if not self.training:
            self.eval()
            self.t5.eval()

        B = len(batch['questions'])  # Note: changed from 'question' to 'questions'
        
        # Convert line_boxes to tensor if it's a list
        if isinstance(batch['line_boxes'], list):
            batch['line_boxes'] = torch.tensor(batch['line_boxes'], dtype=torch.long)
        batch['line_boxes'] = batch['line_boxes'].to(self.device)
        
        # --- 1) Process images ---
        processed_images = self.process_images(batch['images'])
        
        # --- 2) Build text embeddings: question + entity lines ---
        all_sentence_embeddings = []
        all_bboxes = []
        
        for i in range(B):
            # Combine question + lines for this sample
            texts = [batch['questions'][i]] + batch['lines'][i]
            
            # Extract sentence embeddings
            sentence_embeddings = extract_sentence_embeddings(
                texts, self.sent_tokenizer, self.sentence_bert
            )
            
            # Prepare bounding boxes
            orig_w = batch['image_width'][i] if isinstance(batch['image_width'], list) else batch['image_width']
            orig_h = batch['image_height'][i] if isinstance(batch['image_height'], list) else batch['image_height']
            
            # Scale entity boxes
            if len(batch['lines'][i]) > 0:
                ent_boxes = scale_boxes(
                    batch['line_boxes'][i:i+1], orig_w, orig_h
                ).squeeze(0)  # Remove batch dim for single sample
            else:
                ent_boxes = torch.zeros((0, 4), dtype=torch.long, device=self.device)
            
            # Global box for question
            global_bbox = torch.tensor([[0, 0, 512, 512]], dtype=torch.long, device=self.device)
            
            # Concatenate: global + entity boxes
            if len(ent_boxes) > 0:
                boxes = torch.cat([global_bbox, ent_boxes], dim=0)
            else:
                boxes = global_bbox
            
            all_sentence_embeddings.append(torch.from_numpy(sentence_embeddings))
            all_bboxes.append(boxes)
        
        # --- 3) Merge into batches ---
        input_embeds = merge2d(all_sentence_embeddings, 0).to(self.device)
        input_bboxes = merge2d(all_bboxes, 0).to(self.device)
        attention_mask = mask1d(all_sentence_embeddings, 0).to(self.device)
        
        # --- 4) Run GraphDoc encoder ---
        input_data = dict(
            image=processed_images,
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            bbox=input_bboxes,
            return_dict=True
        )
        
        encoder_outputs = self.graphdoc(**input_data)
        all_feats = encoder_outputs.last_hidden_state  # [B, L_total, hidden_size]
        
        # --- 5) Project ALL node embeddings to T5 space ---
        # all_feats: [B, L_total, hidden_size] where L_total = 1 + num_entities
        enc_h = self.proj_to_t5(all_feats)  # [B, L_total, d_model]
        enc_mask = attention_mask
        
        # Initialize return values
        outputs = None
        pred_answers = None
        pred_answer_page = batch.get('answer_page_idx', None)
        pred_answers_conf = None

        if self.training and 'decoder_input_ids' in batch:
            # Training mode
            dec_ids = batch['decoder_input_ids'].to(self.device)
            dec_mask = batch['decoder_attention_mask'].to(self.device)
            
            dec_output = self.t5.decoder(
                input_ids=dec_ids,
                attention_mask=dec_mask,
                encoder_hidden_states=enc_h,
                encoder_attention_mask=enc_mask
            )
            outputs = self.t5.lm_head(dec_output.last_hidden_state)
            
            # Generate predictions during training if requested
            if return_pred_answer:
                with torch.no_grad():
                    pred_answers, pred_answers_conf = self.get_answer_from_model_output(enc_h, enc_mask)
                    
                    # Debug printing during training
                    print("\n=== Training Debug ===")
                    for i in range(min(3, B)):  # Show first 3 examples
                        print(f"Example {i}:")
                        print(f"Question: {batch['questions'][i]}")
                        print(f"Predicted: {pred_answers[i]}")
                        print(f"Ground Truth: {batch['answers'][i]}")
                        print(f"Confidence: {pred_answers_conf[i]:.4f}")
                        print("---")
        else:
            # Inference mode
            pred_answers, pred_answers_conf = self.get_answer_from_model_output(enc_h, enc_mask)
            outputs = BaseModelOutput(last_hidden_state=enc_h)
            
            # Debug printing during inference
            if not self.training:
                print("\n=== Inference Debug ===")
                for i in range(min(3, B)):  # Show first 3 examples
                    print(f"Example {i}:")
                    print(f"Question: {batch['questions'][i]}")
                    print(f"Predicted: {pred_answers[i]}")
                    print(f"Ground Truth: {batch['answers'][i]}")
                    print(f"Confidence: {pred_answers_conf[i]:.4f}")
                    print("---")

        return outputs, pred_answers, pred_answer_page, pred_answers_conf

if __name__ == "__main__":
    # Test the corrected implementation
    from PIL import Image
    
    # B, L_ent, L_dec = 2, 5, 10
    # fake_images = [Image.new("RGB", (1024, 1024)) for _ in range(B)]

    # Initialize model (you'll need to provide correct paths)
    graphdoc_ckpt = "/data2/users/rriccio/pretrained_model/graphdoc"
    sentence_bert_path = "/data2/users/rriccio/pretrained_model/sentence-bert"
    
    B, L_ent, L_dec = 2, 3, 10
    fake_images = [Image.new("RGB",(1024,1024)) for _ in range(B)]

    # Create two identical samples of 3 boxes each:
    two_samples_boxes = torch.tensor([
        [ [100,200,300,400],
        [ 50, 60,100,120],
        [400,500,450,550] ],
        [ [100,200,300,400],
        [ 50, 60,100,120],
        [400,500,450,550] ]
    ], dtype=torch.long)  # shape [2,3,4]

    fake_batch = {
        'questions': ["What is the title?", "How many items?"],
        'lines': [
            ["Title Here","Item 1","Item 2"],
            ["Main Header","List A","List B"]
        ],
        'line_boxes': two_samples_boxes,      # ✅ shape [2,3,4]
        'images': fake_images,
        'image_width': [1024,1024],
        'image_height':[1024,1024],
        'decoder_input_ids': torch.randint(0,32128,(B,L_dec)),
        'decoder_attention_mask': torch.ones(B,L_dec)
    }

    
    
    model = GDOCVQA_ONLYGLOBAL(
        graphdoc_ckpt=graphdoc_ckpt,
        sentence_bert_path=sentence_bert_path,
        t5_name="t5-base",
        device="cuda"
    )
    
    print("Model initialized successfully!")
    # Uncomment to test:
    # model.train()
    # logits = model(fake_batch)
    # print("Training logits shape:", logits.shape)
    
    # model.eval()
    # answers = model(fake_batch)
    # print("Generated answers:", answers)