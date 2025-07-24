# import torch
# from torch.utils.data import DataLoader
# from torchvision.transforms import ToTensor
# # from datasets.InfographicVQA import InfographicsVQADataset, singlepage_docvqa_collate_fn  # adjust import as needed

print("hi")
# from PIL import Image
# from pathlib import Path
# import matplotlib.pyplot as plt

# # 1) Specify your images directory and target save path
# images_dir = Path("/data2/users/rriccio/infographic/infographicsvqa_images")
# target_save_path = Path("/home/rriccio/DocVQA_Project/MP-DocVQA-Framework/70572.png")

# # 2) Locate the file (try common extensions)
# for ext in (".jpeg", ".jpg", ".png"):
#     img_path = images_dir / f"70572{ext}"
#     if img_path.exists():
#         break
# else:
#     raise FileNotFoundError("Could not find 36186.jpeg/.jpg/.png in your images folder")

# # 3) Open and display
# img = Image.open(img_path).convert("RGB")
# plt.figure(figsize=(6,6))
# plt.imshow(img)
# plt.axis("off")
# plt.title(f"Original '70572' ({img.size[0]}×{img.size[1]})")
# plt.show()

# # 4) Save to the specified path
# # Note: If the given directory does not exist, this will raise an error.
# target_save_path.parent.mkdir(parents=True, exist_ok=True)
# img.save(target_save_path)
# print(f"Image has been saved to: {target_save_path}")



# if __name__ == "__main__":
    # config = {
    #     "imdb_dir": "/data2/users/rriccio/infographic/infographicsvqa_qas",
    #     "images_dir": "/data2/users/rriccio/infographic/infographicsvqa_images",
    #     "ocr_dir": "/data2/users/rriccio/infographic/infographicsvqa_ocr"
    # }
    # split = "train"
    # dataset_kwargs = {"ocr_dir": config["ocr_dir"]}
    
    # dataset = InfographicsVQADataset(
    #     imdb_dir=config["imdb_dir"],
    #     images_dir=config["images_dir"],
    #     ocr_dir=config["ocr_dir"],
    #     split=split,
    #     dataset_kwargs=dataset_kwargs,
    #     max_samples=5
    # )

    # #
    # data_loader = DataLoader(dataset, batch_size=2, collate_fn=singlepage_docvqa_collate_fn, shuffle=False)
    
    # for batch in data_loader:
    #     print("\nBatch keys:", batch.keys())
    #     print("Questions:", batch["questions"])
    #     print("Answers:", batch["answers"])
    #     print("Word samples:", batch["words"][0][:5])  # Print first few words of first sample
    #     print("Boxes shape (sample 0):", batch["boxes"][0].shape)
    #     print("Image names:", batch["image_name"])
    #     # If you need to convert images to tensor, you can do:
    #     # transform = ToTensor()
    #     # images = [transform(img) for img in batch["images"]]
    #     break

