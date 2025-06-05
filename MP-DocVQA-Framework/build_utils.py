
'''
CHANGED:
    from models.VT5 import VT5 as VT5
changed:
  dataset = SPDocVQA(config['imdb_dir'], config['images_dir'], split, dataset_kwargsl, max_samples)
changed build?optimizer
'''
import torch
import transformers

from transformers import get_scheduler

# print("=== Running build_utils from:", __file__)


def build_optimizer(model, length_train_loader, config):
    optimizer_class = getattr(transformers, 'AdamW')

    # ✅ Now `.parameters()` works correctly
    optimizer = optimizer_class(model.parameters(), lr=float(config['lr']))

     # --- Print the parameters present in the optimizer ---
    print("=== Parameters in the optimizer ===")
    # Build a mapping of parameter id -> name from the entire model
    param_names = {id(param): name for name, param in model.named_parameters()}
    for group_idx, param_group in enumerate(optimizer.param_groups):
        print(f"Parameter Group {group_idx}:")
        for param in param_group['params']:
            param_name = param_names.get(id(param), "Unknown")
            print(f"  {param_name}: {param.shape}")
            trainable_status = "Trainable" if param.requires_grad else "Frozen"
            print(f"  {param_name}: {param.shape}, {trainable_status}")
    print("=== End of parameters in the optimizer ===")

    num_training_steps = config['train_epochs'] * length_train_loader
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=config['warmup_iterations'], num_training_steps=num_training_steps
    )

    return optimizer, lr_scheduler



# original build?optimizer>
# def build_optimizer(model, length_train_loader, config):
#     optimizer_class = getattr(transformers, 'AdamW')
#     optimizer = optimizer_class(model.model.parameters(), lr=float(config['lr']))
#     num_training_steps = config['train_epochs'] * length_train_loader
#     lr_scheduler = get_scheduler(
#         name="linear", optimizer=optimizer, num_warmup_steps=config['warmup_iterations'], num_training_steps=num_training_steps
#     )

#     return optimizer, lr_scheduler


def build_model(config):

    available_models = ['bertqa', 'longformer', 'bigbird', 'layoutlmv2', 'layoutlmv3', 't5', 'vt5', 'hi-vt5', 'vt5_gdoc', 'vt5_gdoc_2project', 'vt5_gdoc_project_gdoc', 'vt5_gdoc_project_gdoc_trans', 'vt5_gdoc_crossatt', 'vt5_gdoc_nowords', 'vt5_gdoc_addquestion', 'vt5_gdoc_multimodal', 'vt5_gdoc_token', 'vt5_gdoc_weightnowords', 'vt5_gdoc_upsample_and_project', 'vt5_lay_gdoc', 'vt5_gdoc_crossatt_gate', 'vt5_gdoc_mlpfusion', 'vt5_onlysemantic', 'vt5_spatialscaled', 'vt5_lay_visual', 'vt5_longer', 'vt5_freezed', 'graphdoct5vqa']
    if config['model_name'].lower() == 'bert' or config['model_name'].lower() == 'bertqa':
        from models.BertQA import BertQA
        model = BertQA(config)

    elif config['model_name'].lower() == 'longformer':
        from models.Longformer import Longformer
        model = Longformer(config)

    elif config['model_name'].lower() == 'bigbird':
        from models.BigBird import BigBird
        model = BigBird(config)

    elif config['model_name'].lower() == 'layoutlmv2':
        from models.LayoutLMv2 import LayoutLMv2
        model = LayoutLMv2(config)

    elif config['model_name'].lower() == 'layoutlmv3':
        from models.LayoutLMv3 import LayoutLMv3
        model = LayoutLMv3(config)

    elif config['model_name'].lower() == 't5':
        from models.T5 import T5
        model = T5(config)

    elif config['model_name'].lower() == 'vt5':
        from models.VT5 import VT5 as VT5
        model = VT5(config)

    elif config['model_name'].lower() in ['hivt5', 'hi-vt5']:
        from models.HiVT5 import Proxy_HiVT5 as HiVT5
        model = HiVT5(config)

    elif config['model_name'].lower() == 'vt5_gdoc':
        from models.VT5_GDOC import VT5_GDOC as VT5_GDOC
        model = VT5_GDOC(config)
        # dataset_kwargs['use_graphdoc'] = True
        # dataset_kwargs['graphdoc_dir'] = config.get('graphdoc_dir', config['images_dir'])  # Default to images_dir if not specified
    elif config['model_name'].lower() == 'vt5_gdoc_2project':
        from models.VT5_GDOC_2PROJECT import VT5_GDOC_2PROJECT as VT5_GDOC_2PROJECT
        model = VT5_GDOC_2PROJECT(config)
    
    elif config['model_name'].lower() ==  'vt5_gdoc_project_gdoc':
        from models.VT5_GDOC_PROJECT_GDOC import VT5_GDOC_PROJECT_GDOC as VT5_GDOC_PROJECT_GDOC
        model = VT5_GDOC_PROJECT_GDOC(config)
    
    elif config['model_name'].lower() ==  'vt5_gdoc_project_gdoc_trans':
        from models.VT5_GDOC_PROJECT_GDOC_TRANS import VT5_GDOC_PROJECT_GDOC_TRANS as VT5_GDOC_PROJECT_GDOC_TRANS
        model = VT5_GDOC_PROJECT_GDOC_TRANS(config)
    
    elif config['model_name'].lower() ==  'vt5_gdoc_crossatt':
        from models.VT5_GDOC_CROSSATT import VT5_GDOC_CROSSATT as VT5_GDOC_CROSSATT
        model = VT5_GDOC_CROSSATT(config)
    
    elif config['model_name'].lower() ==  'vt5_gdoc_nowords':
        from models.VT5_GDOC_NOWORDS import VT5_GDOC_NOWORDS as VT5_GDOC_NOWORDS
        model = VT5_GDOC_NOWORDS(config)
    
    elif config['model_name'].lower() ==  'vt5_gdoc_addquestion':
        from models.VT5_GDOC_ADDQUESTION import VT5_GDOC_ADDQUESTION as VT5_GDOC_ADDQUESTION
        model = VT5_GDOC_ADDQUESTION(config)
    
    elif config['model_name'].lower() ==  'vt5_gdoc_multimodal':
        from models.VT5_GDOC_MULTIMODAL import VT5_GDOC_MULTIMODAL as VT5_GDOC_MULTIMODAL
        model = VT5_GDOC_MULTIMODAL(config)
    elif config['model_name'].lower() ==  'vt5_gdoc_token':
        from models.VT5_GDOC_TOKEN import VT5_GDOC_TOKEN as VT5_GDOC_TOKEN
        model = VT5_GDOC_TOKEN(config)

    elif config['model_name'].lower() ==  'vt5_gdoc_weightnowords':
        from models.VT5_GDOC_WEIGHTNOWORDS import VT5_GDOC_WEIGHTNOWORDS as VT5_GDOC_WEIGHTNOWORDS
        model = VT5_GDOC_WEIGHTNOWORDS(config)
    
    elif config['model_name'].lower() == 'vt5_gdoc_upsample_and_project':
        from models.VT5_GDOC_UPSAMPLE_AND_PROJECT import VT5_GDOC_UPSAMPLE_AND_PROJECT as VT5_GDOC_UPSAMPLE_AND_PROJECT
        model = VT5_GDOC_UPSAMPLE_AND_PROJECT(config)
    
    elif config['model_name'].lower() == 'vt5_lay_gdoc':
        from models.VT5_LAY_GDOC import VT5_LAY_GDOC as VT5_LAY_GDOC
        model = VT5_LAY_GDOC(config)
    
    elif config['model_name'].lower() == 'vt5_gdoc_crossatt_gate':
        from models.VT5_GDOC_CROSSATT_GATE import VT5_GDOC_CROSSATT_GATE as VT5_GDOC_CROSSATT_GATE
        model = VT5_GDOC_CROSSATT_GATE(config)

    elif config['model_name'].lower() == 'vt5_gdoc_mlpfusion':
        from models.VT5_GDOC_MLPFUSION import VT5_GDOC_MLPFUSION as VT5_GDOC_MLPFUSION
        model = VT5_GDOC_MLPFUSION(config)
    
    elif config['model_name'].lower() == 'vt5_onlysemantic':
        from models.VT5_ONLYSEMANTIC import VT5_ONLYSEMANTIC as VT5_ONLYSEMANTIC
        model = VT5_ONLYSEMANTIC(config)
    
    elif config['model_name'].lower() == 'vt5_spatialscaled':
        from models.VT5_SPATIALSCALED import VT5_SPATIALSCALED as VT5_SPATIALSCALED
        model = VT5_SPATIALSCALED(config)

    elif config['model_name'].lower() == 'vt5_lay_visual':
        from models.VT5_LAY_VISUAL import VT5_LAY_VISUAL as VT5_LAY_VISUAL
        model = VT5_LAY_VISUAL(config)
    elif config['model_name'].lower() == 'vt5_longer':
        from models.VT5_LONGER import VT5_LONGER as VT5_LONGER
        model = VT5_LONGER(config)
    
    elif config['model_name'].lower() == 'vt5_freezed':
        from models.VT5_FREEZED import VT5_FREEZED as VT5_FREEZED
        model = VT5_FREEZED(config)
    
    elif config['model_name'].lower() == 'graphdoct5vqa':
        from models.GRAPHDOCT5VQA import GRAPHDOCT5VQA as GRAPHDOCT5VQA
        model = GRAPHDOCT5VQA(config)
 
 
    else:
        raise ValueError("Value '{:s}' for model selection not expected. Please choose one of {:}".format(config['model_name'], ', '.join(available_models)))

    if config['device'] == 'cuda' and config['data_parallel'] and torch.cuda.device_count() > 1:
        model.parallelize()

    # model.model.to(config['device'])
    model.to(config['device'])
    return model


def build_dataset(config, split, max_samples=None):

    # Specify special params for data processing depending on the model used.
    dataset_kwargs = {}

    if config['model_name'].lower() in ['layoutlmv2', 'layoutlmv3', 'lt5', 'vt5', 'hilt5', 'hi-lt5', 'hivt5', 'hi-vt5', 'vt5_gdoc', 'vt5_gdoc_2project',  'vt5_gdoc_project_gdoc', 'vt5_gdoc_project_gdoc_trans',  'vt5_gdoc_crossatt', 'vt5_gdoc_nowords', 'vt5_gdoc_addquestion', 'vt5_gdoc_multimodal', 'vt5_gdoc_token', 'vt5_gdoc_weightnowords',  'vt5_gdoc_upsample_and_project', 'vt5_lay_gdoc', 'vt5_gdoc_crossatt_gate', 'vt5_gdoc_crossatt_gate', 'vt5_gdoc_mlpfusion', 'vt5_onlysemantic', 'vt5_spatialscaled', 'vt5_lay_visual', 'vt5_longer']:
        dataset_kwargs['get_raw_ocr_data'] = True

    if config['model_name'].lower() in ['layoutlmv2', 'layoutlmv3', 'vt5', 'hivt5', 'hi-vt5', 'vt5_gdoc', 'vt5_gdoc_2project',  'vt5_gdoc_project_gdoc', 'vt5_gdoc_project_gdoc_trans', 'vt5_gdoc_crossatt', 'vt5_gdoc_nowords', 'vt5_gdoc_addquestion', 'vt5_gdoc_multimodal', 'vt5_gdoc_token', 'vt5_gdoc_weightnowords',  'vt5_gdoc_upsample_and_project', 'vt5_lay_gdoc', 'vt5_gdoc_crossatt_gate', 'vt5_gdoc_mlpfusion', 'vt5_onlysemantic', 'vt5_spatialscaled', 'vt5_lay_visual', 'vt5_longer']:
        dataset_kwargs['use_images'] = True

    if config['model_name'].lower() in ['hilt5', 'hi-lt5', 'hivt5', 'hi-vt5']:
        dataset_kwargs['max_pages'] = config.get('max_pages', 1)
        dataset_kwargs['hierarchical_method'] = True
    
    # Additional configuration for GraphDoc if needed
    if config['model_name'].lower() in ['vt5_gvqa', 'vt5_gdoc', 'vt5_gdoc_2project',  'vt5_gdoc_project_gdoc', 'vt5_gdoc_project_gdoc_trans',  'vt5_gdoc_crossatt', 'vt5_gdoc_nowords', 'vt5_gdoc_addquestion', 'vt5_gdoc_multimodal', 'vt5_gdoc_token', 'vt5_gdoc_weightnowords',  'vt5_gdoc_upsample_and_project', 'vt5_lay_gdoc', 'vt5_gdoc_crossatt_gate', 'vt5_gdoc_mlpfusion', 'vt5_spatialscaled', 'vt5_lay_visual', 'vt5_longer']:
        dataset_kwargs['use_graphdoc'] = True  # Enable GraphDoc
        dataset_kwargs['graphdoc_dir'] = config.get('graphdoc_dir', config['images_dir'])

    # Build dataset
    if config['dataset_name'] == 'SP-DocVQA':
        from datasets.SP_DocVQA import SPDocVQA
        dataset = SPDocVQA(config['imdb_dir'], config['images_dir'], split, dataset_kwargs, max_samples)
    
    elif config['dataset_name'] == 'SP-DocVQA_GDOC':  # Add this condition
        from datasets.SP_DocVQA_GDOC import SPDocVQA_GDOC
        # print("max samples in build_utils:", max_samples)
        dataset = SPDocVQA_GDOC(config['imdb_dir'], config['images_dir'], split, dataset_kwargs, max_samples)

    elif config['dataset_name'] == 'SP-DocVQA_LAY_GDOC':  # Add this condition
        from datasets.SP_DocVQA_LAY_GDOC import SPDocVQA_LAY_GDOC
        # print("max samples in build_utils:", max_samples)
        dataset = SPDocVQA_LAY_GDOC(config['imdb_dir'], config['images_dir'], split, dataset_kwargs, max_samples)

    elif config['dataset_name'] == 'MP-DocVQA':
        from datasets.MP_DocVQA import MPDocVQA
        dataset = MPDocVQA(config['imdb_dir'], config['images_dir'], config['page_retrieval'], split, dataset_kwargs)

    elif config['dataset_name'] == 'DUDE':
        from datasets.DUDE import DUDE
        dataset = DUDE(config['imdb_dir'], config['images_dir'], config['page_retrieval'], split, dataset_kwargs)

    elif config['dataset_name'] == 'InfographicVQA':
        from datasets.InfographicVQA import InfographicsVQADataset, singlepage_docvqa_collate_fn
        dataset = InfographicsVQADataset(
            imdb_dir=config["imdb_dir"],
            images_dir=config["images_dir"],
            ocr_dir=config["ocr_dir"],  # explicitly passing OCR directory
            split=split,
            dataset_kwargs=dataset_kwargs,
            max_samples=max_samples
        )
    
    elif config['dataset_name'] == 'InfographicVQA_GRAPHDOC': # <--- ADD THIS NEW BLOCK
        from datasets.InfographicVQA_GRAPHDOC import InfographicsVQADataset, singlepage_docvqa_collate_fn
        dataset = InfographicsVQADataset(
            imdb_dir=config["imdb_dir"],
            images_dir=config["images_dir"],
            ocr_dir=config["ocr_dir"],
            ocr_graphdoc_dir=config["ocr_graphdoc_dir"], # <--- Crucial for GraphDoc
            split=split,
            dataset_kwargs=dataset_kwargs,
            max_samples=max_samples
        )
    else:
        raise ValueError

    return dataset
