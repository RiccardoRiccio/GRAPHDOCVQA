'''
TO CHECK:
- IF EVERUTHING IS IN CORRECT DEVICE
- IF MODEL PACK EVERYTHING (SO IF STUFF IS TRAINABLE)
- UNFROZE PARAMETERS AND LAYERS
- CHECK IF THE .EVAL MAKE SENSE IN self.graphdoc = GraphDocForEncode.from_pretrained(graphdoc_model_path, config=self.gd_cfg).eval()
- mean_pooling: IF IT GETS BATCH AND NOT SINGLE SAMPLE
- CONSIDER ADDING A NORMALIZATION LAYER TO PROJECTOR OR A MLP TO BETTER ALIGNMENT BEFORE PASSING TO T5 DECODER
- CHECK IF QUESTION IS ENCODED AS GLONBAL NODE AND ALSO IF NOT AS A NODE OTHER THEN THE GLOBAL ONE
- check if projection layer is trained and if should go to .eval() at some point since dropout in it
- UNFREEZE GRAPHDOC AND USE DIFFERENT LR FOR T5 AND GDOC
- T5 should automatically do teacher-forcing when labels are passed, but during inference, if you do not use proper decoder_start_token_id, you can get poor generations.
- add dataparallel



'''
import sys
import os
import random
# Ensure parent directory is on PYTHONPATH so we can import model_utils, etc.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import models._model_utils as model_utils

# Path to the GraphDoc_VQA repository (where GraphDoc code lives)
sys.path.append("/home/rriccio/DocVQA_Project/GraphDoc_VQA")

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, T5ForConditionalGeneration, T5Tokenizer
from transformers.modeling_outputs import BaseModelOutput
from layoutlmft.models.graphdoc.configuration_graphdoc import GraphDocConfig
from layoutlmft.models.graphdoc.modeling_graphdoc import GraphDocForEncode


def mean_pooling(model_output, attention_mask):
    """
    Perform mean pooling on token embeddings, ignoring padded tokens.

    Args:
        model_output: Tuple returned by a Hugging Face model. 
                      model_output[0] is a tensor of shape [B, L, H], 
                      where B = batch size, L = sequence length, H = hidden size.
        attention_mask: Tensor of shape [B, L], with 1 for real tokens and 0 for padding.

    Returns:
        pooled: Tensor of shape [B, H], where each row is the average of the valid token embeddings.
    """
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    # print(f"[mean_pooling] token_embeddings.shape = {token_embeddings.shape}")
    # print(f"[mean_pooling] attention_mask.shape = {attention_mask.shape}")
    # print(f"[mean_pooling] attention_mask = {attention_mask}")
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    # print(f"[mean_pooling] output.shape (pooled) = {(torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)).shape}")
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)




def extract_sentence_embeddings(contents, tokenizer, sentence_bert):
    """
    Tokenize a list of sentences, run them through Sentence-BERT, and mean-pool.

    Args:
        contents: List[str] of length B, each a sentence.
        tokenizer: A Hugging Face AutoTokenizer (e.g., for sentence-bert).
        sentence_bert: A Hugging Face AutoModel (Sentence-BERT) on the correct device.

    Steps:
        1. Tokenize `contents` → input_ids, attention_mask of shape [B, L].
        2. Move inputs to sentence_bert.device.
        3. Forward through sentence_bert → model_output[0] of shape [B, L, H].
        4. mean_pooling → pooled embeddings of shape [B, H].
        5. Move to CPU and convert to NumPy → NumPy array of shape (B, H).

    Returns:
        sentence_embeddings: NumPy array of shape (B, H).
    """

    encoded_input = tokenizer(contents, padding=True, truncation=True, return_tensors='pt')
    encoded_input= encoded_input.to(sentence_bert.device)
    # print(f"[extract_sentence_embeddings] tokenized input_ids.shape = {encoded_input['input_ids'].shape}")
    with torch.no_grad():
        model_output = sentence_bert(**encoded_input)
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask']).cpu().numpy()
    # print(f"[extract_sentence_embeddings] sentence_embeddings.shape = {sentence_embeddings.shape}")
    return sentence_embeddings


def merge2d(tensors, pad_id):
    """
    Pad a list of 2D tensors into a single 3D tensor.

    Args:
        tensors: List of B PyTorch tensors, each of shape [L_i, D].
        pad_id: Integer (usually 0) to fill padded positions.

    Steps:
        1. Find L_max = max(L_i) over all tensors.
        2. Find D_max = max(D_i) (usually all D_i are equal).
        3. Create `out` of shape [B, L_max, D_max], filled with pad_id.
        4. For each i, copy tensors[i].shape = [L_i, D_i] into out[i, :L_i, :D_i].

    Returns:
        out: A tensor of shape [B, L_max, D_max], with each original tensor in the first L_i rows,
             padded with pad_id for the remaining rows or columns.
    """
    dim1 = max([s.shape[0] for s in tensors])
    dim2 = max([s.shape[1] for s in tensors])
    out = tensors[0].new(len(tensors), dim1, dim2).fill_(pad_id)
    for i, s in enumerate(tensors):
        out[i, :s.shape[0], :s.shape[1]] = s
    # print(f"[merge2d] out.shape = {out.shape}")
    return out


def mask1d(tensors, pad_id):
    """
    Create a 1D mask for a list of variable-length sequences.

    Args:
        tensors: List of B sequences, each represented by a PyTorch tensor of shape [L_i, ...].
                 (Only len(s) = L_i is used.)
        pad_id: Integer (usually 0) to fill masked/padded positions.

    Steps:
        1. Compute lengths = [L_0, L_1, …, L_{B-1}].
        2. Let L_max = max(lengths).
        3. Create `out` of shape [B, L_max], filled with pad_id (0).
        4. For each i, set out[i, :L_i] = 1 to mark valid positions.

    Returns:
        out: A binary mask tensor of shape [B, L_max], where out[i,j]=1 if j < L_i, else 0.
    """
    lengths= [len(s) for s in tensors]
    out = tensors[0].new(len(tensors), max(lengths)).fill_(pad_id)
    for i, s in enumerate(tensors):
        out[i,:len(s)] = 1
    # print(f"[mask1d] out.shape = {out.shape}")
    return out


class GRAPHDOCT5VQA_ADDWORDS(nn.Module):
    """
    Combines a frozen GraphDoc encoder + a frozen Sentence-BERT for text,
    plus a trainable T5 decoder, to perform Visual Question Answering.

    The dataset + collate_fn are responsible for:
      • resizing each image to [3,512,512]
      • extracting OCR-line texts (as Python lists of strings)
      • scaling and padding line_boxes_rs → [B, Nl_max, 4]
      • building line_mask → [B, Nl_max]
      • grouping everything into a single dict with keys:
          'images', 'questions', 'lines', 'line_boxes_rs', 'line_mask', 'answers' (optional)
    """

    def __init__(self, config):
        super().__init__()

        print("[GRAPHDOCT5VQA __init__] Starting initialization")

        # Pull paths from the config dict
        graphdoc_model_path = config['graphdoc_ckpt']
        sentence_model_path = config['sentence_bert_path']
        t5_model_path = config['t5_name']
        add_projection = config.get('add_projection', True)
        # ──────────── 0) Define Device ────────────
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[GRAPHDOCT5VQA __init__] Using device: {self.device}")

        print(f"[GRAPHDOCT5VQA __init__] graphdoc_model_path = {graphdoc_model_path}")
        print(f"[GRAPHDOCT5VQA __init__] sentence_model_path = {sentence_model_path}")
        print(f"[GRAPHDOCT5VQA __init__] t5_model_path = {t5_model_path}")
        print(f"[GRAPHDOCT5VQA __init__] add_projection = {add_projection}")

        # ──────────── 1) Load + freeze GraphDoc encoder ────────────
        print("[GRAPHDOCT5VQA __init__] Loading GraphDocConfig and GraphDocForEncode…")
        self.gd_cfg = GraphDocConfig.from_pretrained(graphdoc_model_path)
        self.graphdoc = GraphDocForEncode.from_pretrained(graphdoc_model_path, config=self.gd_cfg)
        num_graphdoc_params = sum(p.numel() for p in self.graphdoc.parameters())
        print(f"[GRAPHDOCT5VQA __init__] Loaded GraphDoc; #params = {num_graphdoc_params:,}")

        # DEBUG: list all public attributes in GraphDocConfig
        cfg_fields = [n for n in dir(self.gd_cfg) if not n.startswith("_")]
        print(f"[GRAPHDOCT5VQA __init__] GraphDocConfig fields:\n  {cfg_fields}")

        # for p in self.graphdoc.parameters():
        #     p.requires_grad = False
        # print("[GRAPHDOCT5VQA __init__] Frozen all GraphDoc parameters")


        # ──────────── 2) Load + freeze Sentence-BERT (for questions & lines) ────────────
        print("[GRAPHDOCT5VQA __init__] Loading Sentence-BERT tokenizer and model…")
        self.sent_tokenizer = AutoTokenizer.from_pretrained(sentence_model_path)
        self.sentence_bert = AutoModel.from_pretrained(sentence_model_path).eval()
        num_sbert_params = sum(p.numel() for p in self.sentence_bert.parameters())
        print(f"[GRAPHDOCT5VQA __init__] Loaded Sentence-BERT; #params = {num_sbert_params:,}")
        for p in self.sentence_bert.parameters():
            p.requires_grad = False
        print("[GRAPHDOCT5VQA __init__] Frozen all Sentence-BERT parameters")

        # ──────────── 3) Load T5 for answer generation ────────────
        print("[GRAPHDOCT5VQA __init__] Loading T5 tokenizer and model…")
        self.t5_tokenizer = T5Tokenizer.from_pretrained(t5_model_path)
        self.t5 = T5ForConditionalGeneration.from_pretrained(t5_model_path)
        num_t5_params = sum(p.numel() for p in self.t5.parameters())
        print(f"[GRAPHDOCT5VQA __init__] Loaded T5; #params = {num_t5_params:,}")
        # Freeze T5’s encoder; only train decoder + lm_head
        for p in self.t5.encoder.parameters():
            p.requires_grad = False
        print("[GRAPHDOCT5VQA __init__] Frozen T5 encoder parameters; decoder+lm_head remain trainable")

        for p in self.t5.shared.parameters():
            p.requires_grad = True
        print("[GRAPHDOCT5VQA __init__] Unfrozen T5 shared embedding layer")
        

        # ──────────── 4) Optional projection from GraphDoc hidden_size → T5 d_model ────────────
     
        self.gd_hidden = self.gd_cfg.hidden_size      # GraphDoc embedding size
        self.t5_hidden = self.t5.config.d_model       # T5 embedding size
        print(f"[GRAPHDOCT5VQA __init__] GraphDoc hidden_size = {self.gd_hidden}, T5 d_model = {self.t5_hidden}")
        self.add_projection = add_projection
        if self.add_projection:
            # self.projection = nn.Linear(self.gd_hidden, self.t5_hidden)
            self.projection = nn.Sequential(
                nn.Linear(self.gd_hidden, self.t5_hidden),
                nn.LayerNorm(self.t5_hidden),      # Optional: stabilize training
                nn.ReLU(),                         # Optional: nonlinearity
                nn.Dropout(0.1),                   # Optional: regularization
            )
            self.projection.to(self.device)
            print("[GRAPHDOCT5VQA __init__] Added linear projection layer from GraphDoc→T5.")


        # ──────────── 5) Device placement ────────────
        self.graphdoc.to(self.device)
        self.sentence_bert.to(self.device)
        self.t5.to(self.device)

        print("[GRAPHDOCT5VQA __init__] Finished initialization\n")

    def encode_question(self, question: str) -> torch.Tensor:
        """
        Encode a single question string (or list of one string) via Sentence-BERT.
        Returns [1, hidden_size] on `self.device`.
        """
        q_list = [question] if isinstance(question, str) else question
        # print(f"[encode_question] Encoding question: {q_list}")
        enc = self.sent_tokenizer(q_list, padding=True, truncation=True, return_tensors="pt")
        enc = {k: v.to(self.device) for k, v in enc.items()}
        # print(f"[encode_question] tokenized input_ids.shape = {enc['input_ids'].shape}")
        with torch.no_grad():
            out = self.sentence_bert(**enc)
        pooled = mean_pooling(out, enc["attention_mask"])  # [batch=1, hidden_size]
        # print(f"[encode_question] pooled.shape = {pooled.shape}")
        return pooled  # Already on self.device

    def process_batch_data(self, batch_data: dict) -> dict:
        """
        Build GraphDoc inputs – this time every node (question + words + lines)
        has both an embedding and a bbox, so the two sequence-lengths always match.
        """
        device = next(self.parameters()).device   # device where the model lives


        B      = len(batch_data["questions"])

        # ── move fixed-shSape tensors once ─────────────────────────────────────────
        images     = batch_data["images"].to(device).float()           # [B,3,512,512]
        word_boxes = batch_data["word_boxes_resized"].to(device)       # [B,Nw_max,4]  (padded)
        line_boxes = batch_data["line_boxes_rs"].to(device)            # [B,Nl_max,4]  (padded)
        # ──────────────────────────────────────────────────────────────────────────

        all_node_embeds  = []
        all_node_bboxes  = []

        for i in range(B):
            # 1) question → CLS node
            q_emb   = self.encode_question(batch_data["questions"][i])      # [1,H]
            q_emb   = q_emb.to(device)            
            cls_box = torch.tensor([0,0,512,512], dtype=torch.long, device=device)

            # 2) words
            words            = batch_data["words"][i]          # real list[str]
            n_w              = len(words)
            w_boxes_real     = word_boxes[i, :n_w]             # trim padding → on CUDA
            if n_w:
                w_np   = extract_sentence_embeddings(words, self.sent_tokenizer, self.sentence_bert)
                w_emb  = torch.from_numpy(w_np).to(device)     # [n_w,H]
            else:
                w_emb  = torch.zeros((0, self.gd_hidden), device=device)

            # 3) lines
            lines            = batch_data["lines"][i]
            n_l              = len(lines)
            l_boxes_real     = line_boxes[i, :n_l]             # trim padding
            if n_l:
                l_np   = extract_sentence_embeddings(lines, self.sent_tokenizer, self.sentence_bert)
                l_emb  = torch.from_numpy(l_np).to(device)     # [n_l,H]
            else:
                l_emb  = torch.zeros((0, self.gd_hidden), device=device)

            # 4) concat (CLS + words + lines)  → boxes & embeds have SAME length
            boxes  = torch.cat([cls_box.unsqueeze(0), w_boxes_real, l_boxes_real], dim=0)
            embeds = torch.cat([q_emb,               w_emb,        l_emb       ], dim=0)

            all_node_bboxes.append(boxes)
            all_node_embeds.append(embeds)

        # ── pad across the batch ────────────────────────────────────────────────
        inputs_embeds  = merge2d(all_node_embeds, pad_id=0).to(device)   # [B,L_max,H]
        attention_mask = mask1d(all_node_embeds, pad_id=0).to(device)    # [B,L_max]

        max_len = max(b.shape[0] for b in all_node_bboxes)
        padded_boxes = torch.zeros((B, max_len, 4), dtype=torch.long, device=device)
        for idx, b in enumerate(all_node_bboxes):
            padded_boxes[idx, :b.shape[0]] = b
        # ─────────────────────────────────────────────────────────────────────────

        return {
            "image":          images,
            "inputs_embeds":  inputs_embeds,
            "attention_mask": attention_mask,
            "bbox":           padded_boxes,
            "return_dict":    True,
        }





    def forward(
        self,
        batch_data: dict,
        # target_answers: list = None,
        return_pred_answer: bool = False
    ):
        """
        If `self.training == True` and `target_answers is None`:
          • automatically grab `target_answers = batch_data["answers"]` → TRAINING mode
          • Tokenize `target_answers` via T5 tokenizer → labels
          • Run T5 with `labels=…` → compute `loss`, `logits` → store in `t5_out`
          • If `return_pred_answer=True`, also run `generate(...)` to obtain predictions + confidences
          → return `(t5_out, pred_texts, None, confidences)`

        If `self.training == False` (eval mode):
          • Run T5.generate(...) to get predictions + confidences
          • If `return_pred_answer=True`, return `(None, pred_texts, None, confidences)`
          • Otherwise return a dict with `"pred_answers", "pred_confidences", ...`
        """
        device = batch_data["images"].device
        # print("\n[GRAPHDOCT5VQA forward] Entered forward()")
        # 1) If in training mode but no `target_answers` passed, fetch from batch_data:

        # 2) Build GraphDoc input → run the frozen GraphDoc encoder
        # print("[forward] Calling process_batch_data(...)")
        graphdoc_input = self.process_batch_data(batch_data)
        # print("[forward] Running GraphDoc encoder…")

        # ← INSERT HERE: inspect exactly what will be passed to self.graphdoc
        # img_tensor = graphdoc_input["image"]
        # bboxes    = graphdoc_input["bbox"]
        # print(f"DEBUG [forward → calling graphdoc]: image.shape = {img_tensor.shape}, device = {img_tensor.device}")
        # print(f"DEBUG [forward → calling graphdoc]: bbox.shape = {bboxes.shape}, "
        #       f"min = {bboxes.min().item()}, max = {bboxes.max().item()}")
        encoder_output = self.graphdoc(**graphdoc_input)
        node_embeds = encoder_output.last_hidden_state  # [B, L_tot, H_sent]
        # print(f"[forward] GraphDoc encoder_output.last_hidden_state.shape = {node_embeds.shape}")


        # 3) Optionally project GraphDoc embeddings to T5’s d_model
        if self.add_projection:
            # print(f"[forward] Applying projection: {node_embeds.shape} → ", end="")
            node_embeds = self.projection(node_embeds)  # [B, L_tot, d_model]
            print(f"{node_embeds.shape}")

        enc_attn_mask = graphdoc_input["attention_mask"]  # [B, L_tot]
        # print(f"[forward] GraphDoc attention_mask.shape = {enc_attn_mask.shape}")


        # ──────── TRAINING BRANCH ────────
        if self.training:
            print("[forward] Detected target_answers → Training branch")
            # Randomly select one answer per instance
            target_answers = [random.choice(ans_list) for ans_list in batch_data["answers"]]
            # print(f"[forward] Training mode – fetched target_answers = {target_answers}")


        
            # 3a) Tokenize the gold answers (teacher‐forcing)
            # print(f"[forward] Tokenizing target_answers = {target_answers}")
            tgt = self.t5_tokenizer(
                target_answers, padding=True, truncation=True, return_tensors="pt"
            )
            labels = tgt["input_ids"]
            labels = labels.to(self.device)         # [B, L_tgt]
            # Mask out pad tokens so T5 ignores them in the loss:
            labels[labels == self.t5_tokenizer.pad_token_id] = -100
            # tgt_mask = tgt["attention_mask"].to(self.device)   # [B, L_tgt]
            
            # print(f"[forward] tgt_ids.shape = {labels.shape}")
            # print(" tgt_ids ",  labels)


            # 3b) Prepare decoder_input_ids
            # decoder_input_ids = self.t5._shift_right(labels)# [B, L_tgt]
            # print(f"[forward] decoder_input_ids.shape = {decoder_input_ids.shape}")
            # print(f"[forward] decoder_input_ids = {decoder_input_ids}")

            # 3c) Wrap node_embeds in BaseModelOutput to feed as T5’s encoder_outputs
            enc_out_for_t5 = BaseModelOutput(last_hidden_state=node_embeds)
            # print("[forward] Created BaseModelOutput for T5 encoder_outputs")

            # print("[forward] Running T5 with labels to compute loss and logits…")

            # 3d) Call T5 with labels to compute loss & logits
            t5_out = self.t5(
                encoder_outputs=enc_out_for_t5,
                attention_mask=enc_attn_mask,
                # decoder_input_ids=decoder_input_ids,
                # decoder_attention_mask=tgt_mask,
                labels=labels,
                return_dict=True
            )
            # print(f"[forward] T5 returned loss = {t5_out.loss.item()}, logits.shape = {t5_out.logits.shape}")
            # if return_pred_answer:

            #     # print("[forward] return_pred_answer=True, generating predictions for logging…")
            #     # After computing the loss, also generate predicted answers for logging
            #     with torch.no_grad():
            #         # print("tojen id.....", self.t5.config.decoder_start_token_id)

            #         gen_out = self.t5.generate(
            #             encoder_outputs=enc_out_for_t5,
            #             attention_mask=enc_attn_mask,
            #             decoder_start_token_id=self.t5.config.decoder_start_token_id,
            #             max_length=32,
            #             num_beams=4,
            #             early_stopping=True,
            #             return_dict_in_generate=True,
            #             output_scores=True
            #         )
            #     pred_ids = gen_out.sequences
            #     pred_texts = self.t5_tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
            #     print(f"[forward] Generated pred_texts = {pred_texts}")
            #     try:
            #         confidences = model_utils.get_generative_confidence(gen_out)
            #     except:
            #         confidences = [0.0] * len(pred_texts)
            #     print(f"[forward] Generated confidences = {confidences}")

            #     return t5_out, pred_texts, None, confidences

            # else:
            #     print("[forward] Returning training‐mode dict (no pred_answers)")
            #     return {
            #         "loss": t5_out.loss,
            #         "logits": t5_out.logits,            # [B, L_tgt, vocab_size]
            #         "encoder_output": encoder_output,
            #         "node_embeddings": node_embeds
            #     }
            print("[forward] Returning training‐mode dict (no pred_answers)")
            return {
                "loss": t5_out.loss,
                "logits": t5_out.logits,            # [B, L_tgt, vocab_size]
                "encoder_output": encoder_output,
                "node_embeddings": node_embeds
            }

        # ──────── EVALUATION / INFERENCE BRANCH ────────
        else:
            print("[forward] No target_answers → Inference branch")
            # print("self.training", self.training)
            enc_out_for_t5 = BaseModelOutput(last_hidden_state=node_embeds)
            # print("[forward] Running T5.generate(...)")
            gen_out = self.t5.generate(
                encoder_outputs=enc_out_for_t5,
                attention_mask=enc_attn_mask,
                decoder_start_token_id=self.t5.config.decoder_start_token_id, 
                max_length=32,
                num_beams=4,
                early_stopping=True,
                return_dict_in_generate=True,
                output_scores=True
            )
            pred_ids = gen_out.sequences           # [B, L_gen]
            # print("Predicted token IDs:", pred_ids)  # <--- 🔍 Add this line here
            pred_texts = self.t5_tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
            # print(f"[forward] Generated pred_texts = {pred_texts}")
            try:
                confidences = model_utils.get_generative_confidence(gen_out)
            except:
                confidences = [0.0] * len(pred_texts)
            # print(f"[forward] Generated confidences = {confidences}")

            # Debug‐print first two examples (optional, but matches prior behavior)
            if not self.training:
                print("\n=== Inference Debug (FIRST 2 OF THE BATCH) ===")
                for i in range(min(2, len(pred_texts))):
                    print(f"Q: {batch_data['questions'][i]}")
                    print(f"→ Predicted: {pred_texts[i]}")
                    print(f"→ Confidence: {confidences[i]:.4f}")
                    print(f"→ Ground Truth: {batch_data.get('answers',[[]])[i]}")
                    print("-----------")

            if return_pred_answer:
                print("[forward] return_pred_answer=True → returning pred_texts and confidences")
                return None, pred_texts, None, confidences
            else:
                # print("[forward] Returning inference‐mode dict")
                return {
                    "pred_answers": pred_texts,
                    "pred_confidences": confidences,
                    "encoder_output": encoder_output,
                    "node_embeddings": node_embeds,
                    "encoder_attention_mask": enc_attn_mask
                }


# def create_model(config) -> GRAPHDOCT5VQA:
#     """
#     Factory function: returns a GRAPHDOCT5VQA on GPU (if available).
#     This is what `build_model(config)` will ultimately call.
#     """
#     model = GRAPHDOCT5VQA(config)
#     return model


# # ────────────────────────────────────────────────────────────────────────────────
# # Quick sanity‐check (run as script)
# # ────────────────────────────────────────────────────────────────────────────────
# if __name__ == "__main__":
#     import torch
#     from torch.utils.data import DataLoader
#     from datasets.InfographicVQA_GRAPHDOC import InfographicsVQADataset, singlepage_docvqa_collate_fn


#     print("[main] Starting sanity check")

#     # Example config (update paths as necessary)
#     config = {
#         "graphdoc_ckpt": "/data2/users/rriccio/pretrained_model/graphdoc",
#         "sentence_bert_path": "/data2/users/rriccio/pretrained_model/sentence-bert",
#         "t5_name": "t5-base",
#         "add_projection": True,
#     }

#     # Create dataset & DataLoader (for debug)
#     dataset = InfographicsVQADataset(
#         imdb_dir="/data2/users/rriccio/infographic/infographicsvqa_qas",
#         images_dir="/data2/users/rriccio/infographic/infographicsvqa_images",
#         ocr_dir="/data2/users/rriccio/infographic/infographicsvqa_ocr",
#         ocr_graphdoc_dir="/data2/users/rriccio/easyocr_infographic",
#         split="train",
#         dataset_kwargs={"ocr_dir": "/data2/users/rriccio/infographic/infographicsvqa_ocr"},
#         max_samples=4
#     )
#     loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=singlepage_docvqa_collate_fn)

#     # Instantiate model
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = GRAPHDOCT5VQA(config).to(device)

#     # SANITY‐CHECK #1: One training‐style forward (with teacher forcing)
#     model.train()
#     for batch in loader:
#         # Move tensors to GPU
#         batch["images"] = batch["images"].to(device)
#         batch["line_boxes_rs"] = batch["line_boxes_rs"].to(device)
#         batch["line_mask"] = batch["line_mask"].to(device)

#         # Call forward without passing target_answers explicitly:
#         outputs, pred_answers, _, confidences = model.forward(batch, return_pred_answer=True)
#         # print("Batch loss:", outputs.loss.item())
#         # print("Predictions:", pred_answers)
#         # print("Confidences:", confidences)
#         break

#     # SANITY‐CHECK #2: One inference‐style forward
#     model.eval()
#     with torch.no_grad():
#         for batch in loader:
#             batch["images"] = batch["images"].to(device)
#             batch["line_boxes_rs"] = batch["line_boxes_rs"].to(device)
#             batch["line_mask"] = batch["line_mask"].to(device)

#             _, pred_answers, _, confidences = model.forward(batch, return_pred_answer=True)
#             print("Eval Predictions:", pred_answers)
#             print("Eval Confidences:", confidences)
#             break
