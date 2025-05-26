import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from datasets.InfographicVQA import InfographicsVQADataset, singlepage_docvqa_collate_fn  # adjust import as needed

if __name__ == "__main__":
    config = {
        "imdb_dir": "/data2/users/rriccio/infographic/infographicsvqa_qas",
        "images_dir": "/data2/users/rriccio/infographic/infographicsvqa_images",
        "ocr_dir": "/data2/users/rriccio/infographic/infographicsvqa_ocr"
    }
    split = "train"
    dataset_kwargs = {"ocr_dir": config["ocr_dir"]}
    
    dataset = InfographicsVQADataset(
        imdb_dir=config["imdb_dir"],
        images_dir=config["images_dir"],
        ocr_dir=config["ocr_dir"],
        split=split,
        dataset_kwargs=dataset_kwargs,
        max_samples=5
    )

    #
    data_loader = DataLoader(dataset, batch_size=2, collate_fn=singlepage_docvqa_collate_fn, shuffle=False)
    
    for batch in data_loader:
        print("\nBatch keys:", batch.keys())
        print("Questions:", batch["questions"])
        print("Answers:", batch["answers"])
        print("Word samples:", batch["words"][0][:5])  # Print first few words of first sample
        print("Boxes shape (sample 0):", batch["boxes"][0].shape)
        print("Image names:", batch["image_name"])
        # If you need to convert images to tensor, you can do:
        # transform = ToTensor()
        # images = [transform(img) for img in batch["images"]]
        break
