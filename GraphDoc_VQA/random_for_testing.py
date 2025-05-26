import os
import sys
import json
import torch
from pathlib import Path
from torch.utils.data import DataLoader
import yaml
from tqdm import tqdm

# Add the parent directory to the Python path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

# Import local modules (assuming same structure as training script)
from samples.metrics import Evaluator
from samples.finetune_vt5graphdoc import (
    HybridVT5GraphDoc,
    DocVQAQuestionDrivenDataset,
    docvqa_collate_fn,
)

def load_config(config_file):
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)

def evaluate_model(model, val_loader, evaluator, num_examples=20):
    model.vt5.model.eval()
    all_predictions = []
    all_ground_truth = []
    all_questions = []
    all_image_names = []
    
    print("\nEvaluating model...")
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluation"):
            outputs, pred_answers, _ = model.forward(
                batch, 
                return_pred_answer=True,
                return_confidence=False
            )
            
            # Store predictions and ground truth
            all_predictions.extend(pred_answers)
            all_ground_truth.extend(batch["all_answers"])
            all_questions.extend(batch["questions"])
            all_image_names.extend(batch["image_name"])

    # Calculate overall metrics
    metrics = evaluator.get_metrics(all_ground_truth, all_predictions)
    mean_accuracy = float(sum(metrics["accuracy"]) / len(metrics["accuracy"]))
    mean_anls = float(sum(metrics["anls"]) / len(metrics["anls"]))
    
    print("\n=== Overall Evaluation Metrics ===")
    print(f"Mean Accuracy: {mean_accuracy:.4f}")
    print(f"Mean ANLS: {mean_anls:.4f}")
    
    print("\n=== Sample Predictions (20 examples) ===")
    print("Format: Question | Prediction | Ground Truth | Image Name")
    print("-" * 80)
    
    for idx in range(min(num_examples, len(all_predictions))):
        question = all_questions[idx]
        prediction = all_predictions[idx]
        ground_truth = all_ground_truth[idx]
        image_name = all_image_names[idx]
        
        print(f"\nExample {idx + 1}:")
        print(f"Question: {question}")
        print(f"Prediction: {prediction}")
        print(f"Ground Truth: {ground_truth}")
        print(f"Image: {image_name}")
        print(f"Accuracy: {metrics['accuracy'][idx]:.4f}")
        print(f"ANLS: {metrics['anls'][idx]:.4f}")
        print("-" * 40)

def main():
    # Load configuration
    config_file = "config/models/vt5.yml"
    config = load_config(config_file)
    
    # Initialize dataset and dataloader
    val_dataset = DocVQAQuestionDrivenDataset(config, split="val")
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=docvqa_collate_fn
    )
    
    # Initialize model
    model = HybridVT5GraphDoc(config)
    
    # Load trained weights
    weights_path = Path(config["save_dir"])
    if not weights_path.exists():
        raise ValueError(f"Weights directory not found: {weights_path}")
    
    print(f"Loading model weights from: {weights_path}")
    model.vt5.model.load_state_dict(torch.load(weights_path / "pytorch_model.bin"))
    
    # Move model to device
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    model.vt5.to(device)
    
    # Initialize evaluator
    evaluator = Evaluator()
    
    # Run evaluation
    evaluate_model(model, val_loader, evaluator)

if __name__ == "__main__":
    main()