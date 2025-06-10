'''
CHNAGED:
    # model.model.save_pretrained(os.path.join(save_dir, "model__{:d}.ckpt".format(epoch)))
to:
    if hasattr(model, "save_pretrained"):
    # If the top‐level model is itself a HF PreTrainedModel:
        model.save_pretrained(os.path.join(save_dir, "model__{:d}.ckpt".format(epoch)))
    elif hasattr(model, "model"):
        # Fallback to the wrapped “.model” object
        model.model.save_pretrained(os.path.join(save_dir, "model__{:d}.ckpt".format(epoch)))
    else:
        # As a last resort, just dump the state_dict()
        torch.save(model.state_dict(), os.path.join(save_dir, "model__{:d}.pt".format(epoch)))

CHANGED:
    ADDED: os.makedirs(save_root, exist_ok=True)
'''

# import os
# from utils import save_yaml
# import torch

# def save_model(model, epoch, update_best=False, **kwargs):
#     save_dir = os.path.join(kwargs['save_dir'], 'checkpoints', "{:s}_{:s}_{:s}".format(kwargs['model_name'].lower(), kwargs.get('page_retrieval', '').lower(), kwargs['dataset_name'].lower()))
#     # 2) Make sure it exists
#     os.makedirs(save_dir, exist_ok=True)
#     # model.model.save_pretrained(os.path.join(save_dir, "model__{:d}.ckpt".format(epoch)))
#     if hasattr(model, "save_pretrained"):
#     # If the top‐level model is itself a HF PreTrainedModel:
#         print("Saving model as itself a HF PreTrainedModel | model.save_pretrained(etc)")
#         model.save_pretrained(os.path.join(save_dir, "model__{:d}.ckpt".format(epoch)))
#     elif hasattr(model, "model"):
#         # Fallback to the wrapped “.model” object
#         print("Saving model as wrapped “.model” object | model.model.save_pretrained(etc)")
#         model.model.save_pretrained(os.path.join(save_dir, "model__{:d}.ckpt".format(epoch)))
#     else:
#         # As a last resort, just dump the state_dict()
#         print("Saving model as state_dict() | torch.save(model.state_dict(), os.path.join(etc)")
#         torch.save(model.state_dict(), os.path.join(save_dir, "model__{:d}.pt".format(epoch)))


#     tokenizer = model.tokenizer if hasattr(model, 'tokenizer') else model.processor if hasattr(model, 'processor') else None
#     if tokenizer is not None:
#         tokenizer.save_pretrained(os.path.join(save_dir, "model__{:d}.ckpt".format(epoch)))

#     if hasattr(model.model, 'visual_embeddings'):
#         model.model.visual_embeddings.feature_extractor.save_pretrained(os.path.join(save_dir, "model__{:d}.ckpt".format(epoch)))

#     save_yaml(os.path.join(save_dir, "model__{:d}.ckpt".format(epoch), "experiment_config.yml"), kwargs)

#     if update_best:
#         model.model.save_pretrained(os.path.join(save_dir, "best.ckpt"))
#         tokenizer.save_pretrained(os.path.join(save_dir, "best.ckpt"))
#         save_yaml(os.path.join(save_dir, "best.ckpt", "experiment_config.yml"), kwargs)


# def load_model(base_model, ckpt_name, **kwargs):
#     load_dir = kwargs['save_dir']
#     base_model.model.from_pretrained(os.path.join(load_dir, ckpt_name))

import os
from utils import save_yaml
import torch

def save_model(model, epoch, update_best=False, **kwargs):
    """
    Saves the model checkpoints according to whether the model is:
      1. A HuggingFace PreTrainedModel itself (has save_pretrained),
      2. A wrapper around a sub‐module (has model), or
      3. Neither (fallback to state_dict()).

    Regardless of which branch is taken, we ensure that the directory
    “…/model__{epoch}.ckpt/” exists before attempting to write experiment_config.yml inside it.
    """
    # Build the path: …/save_dir/checkpoints/<model>_<page_retrieval>_<dataset>/
    save_root = os.path.join(
        kwargs['save_dir'],
        'checkpoints',
        "{:s}_{:s}_{:s}".format(
            kwargs['model_name'].lower(),
            kwargs.get('page_retrieval', '').lower(),
            kwargs['dataset_name'].lower()
        )
    )
    # 1) Make sure the checkpoint directory exists
    os.makedirs(save_root, exist_ok=True)

    # Decide how to save the model weights
    if hasattr(model, "save_pretrained"):
        # Case A: The top‐level model is itself an HF PreTrainedModel
        # e.g., model = T5ForConditionalGeneration.from_pretrained(...)
        print("Saving as HF PreTrainedModel via model.save_pretrained(...)")
        model.save_pretrained(os.path.join(save_root, f"model__{epoch}.ckpt"))
    elif hasattr(model, "model"):
        # Case B: The actual HF model is wrapped inside model.model
        print("Saving wrapped sub‐module via model.model.save_pretrained(...)")
        model.model.save_pretrained(os.path.join(save_root, f"model__{epoch}.ckpt"))
    else:
        # Case C: Fallback to state_dict()
        print("Saving state_dict() fallback")
        torch.save(
            model.state_dict(),
            os.path.join(save_root, f"model__{epoch}.pt")
        )

    # 2) Regardless of which branch, ensure “model__{epoch}.ckpt/” exists before writing YAML
    ckpt_dir = os.path.join(save_root, f"model__{epoch}.ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)

    # 3) Save the tokenizer/processor if present
    tokenizer = (
        model.tokenizer
        if hasattr(model, 'tokenizer')
        else model.processor
        if hasattr(model, 'processor')
        else None
    )
    if tokenizer is not None:
        # This will create files like
        # …/model__{epoch}.ckpt/config.json, tokenizer.json, etc.
        tokenizer.save_pretrained(ckpt_dir)

    # 4) If the wrapped HF model has a visual_embeddings sub‐module, save its feature extractor
    if hasattr(model, "model") and hasattr(model.model, 'visual_embeddings'):
        model.model.visual_embeddings.feature_extractor.save_pretrained(ckpt_dir)

    # 5) Write out experiment_config.yml inside “model__{epoch}.ckpt/”
    save_yaml(os.path.join(ckpt_dir, "experiment_config.yml"), kwargs)

    # 6) If this checkpoint is marked “update_best,” also overwrite “best.ckpt/”
    if update_best:
        best_ckpt_dir = os.path.join(save_root, "best.ckpt")
        os.makedirs(best_ckpt_dir, exist_ok=True)

        if hasattr(model, "save_pretrained"):
            model.save_pretrained(best_ckpt_dir)
        elif hasattr(model, "model"):
            model.model.save_pretrained(best_ckpt_dir)
        else:
            torch.save(model.state_dict(), os.path.join(best_ckpt_dir, "model_state.pt"))

        if tokenizer is not None:
            tokenizer.save_pretrained(best_ckpt_dir)

        if hasattr(model, "model") and hasattr(model.model, 'visual_embeddings'):
            model.model.visual_embeddings.feature_extractor.save_pretrained(best_ckpt_dir)

        save_yaml(os.path.join(best_ckpt_dir, "experiment_config.yml"), kwargs)


def load_model(base_model, ckpt_name, **kwargs):
    """
    Loads weights into base_model (which is assumed to wrap a .model sub‐module)
    from a directory or file named `ckpt_name` inside kwargs['save_dir'].

    Usage example:
        base_model = GRAPHDOCT5VQA(config)
        load_model(base_model, "model__3.ckpt", save_dir="/path/to/save_root")
    """
    load_root = os.path.join(kwargs['save_dir'], 'checkpoints')
    full_path = os.path.join(load_root, ckpt_name)

    if os.path.isdir(full_path) and hasattr(base_model, "model"):
        # If ckpt_name points to a directory produced by save_pretrained(), use from_pretrained
        base_model.model.from_pretrained(full_path)
    elif full_path.endswith(".pt") or full_path.endswith(".pth"):
        # If ckpt_name is a .pt file, load state_dict()
        state_dict = torch.load(full_path, map_location=kwargs.get('device', 'cpu'))
        base_model.load_state_dict(state_dict)
    else:
        raise ValueError(f"No valid checkpoint found at {full_path}")
