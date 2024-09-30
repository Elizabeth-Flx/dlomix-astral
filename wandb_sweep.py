import os

import numpy as np
import pandas as pd
import tensorflow as tf

from dlomix.losses import masked_spectral_distance, masked_pearson_correlation_distance
import yaml

MAX_SEQUENCE_LENGTH = 30
MAX_COLLISION_ENERGY = 6

protein_map = {
    'A': 1,
    'C': 2,  # fixed modification cysteine
    'D': 3,
    'E': 4,
    'F': 5,
    'G': 6,
    'H': 7,
    'I': 8,
    'K': 9,
    'L': 10,
    'M': 11,
    'N': 12,
    'P': 13,
    'Q': 14,
    'R': 15,
    'S': 16,
    'T': 17,
    'V': 18,
    'W': 19,
    'Y': 20,
    'm': 21, # oxidized methionine
}

print('='*32)
print('Conda info')
print(f"Environment: {os.environ['CONDA_DEFAULT_ENV']}")
print('='*32)
print('Tensorflow info')
print("TensorFlow version:", tf.__version__)
print(f"Built with CUDA: {tf.test.is_built_with_cuda()}")
print(f"Number of GPUs available: {len(tf.config.list_physical_devices('GPU'))}")
print(f"List of GPUs available: {tf.config.list_physical_devices('GPU')}")
print('='*32)


with open("/nfs/home/students/d.lochert/projects/astral/dlomix-astral/config.yaml", 'r') as yaml_file:
    config = yaml.safe_load(yaml_file)

model_settings = config['model_settings']
train_settings = config['train_settings']

print("DataLoader Settings:")
print(f"Dataset: {config['dataloader']['dataset']}")
print(f"Batch Size: {config['dataloader']['batch_size']}")

if config['model_type'] == 'ours':
    print("\nOur Model config:")
    for key, value in model_settings.items():
        print(f"{key}: {value}")
elif config['model_type'] == 'prosit_t':
    print("\nProsit_t Model config:")
    for key, value in config['prosit'].items():
        print(f"{key}: {value}")

print("\nTraining Settings:")
for key, value in train_settings.items():
    print(f"{key}: {value}")
print('='*32)



###################################################
#                   Data Loader                   #
###################################################

from dlomix.data import FragmentIonIntensityDataset
from dlomix.data import load_processed_dataset

os.environ['HF_HOME'] = "/cmnfs/proj/prosit_astral"
os.environ['HF_DATASETS_CACHE'] = "/cmnfs/proj/prosit_astral/datasets"

CUSTOM_ALPHABET = {
    'A': 1,
    'C': 2,
    'D': 3,
    'E': 4,
    'F': 5,
    'G': 6,
    'H': 7,
    'I': 8,
    'K': 9,
    'L': 10,
    'M': 11,
    'N': 12,
    'P': 13,
    'Q': 14,
    'R': 15,
    'S': 16,
    'T': 17,
    'V': 18,
    'W': 19,
    'Y': 20,
    'm': 21, # oxidized methionine
}

os.environ['HF_HOME'] = "/cmnfs/proj/prosit_astral"
os.environ['HF_DATASETS_CACHE'] = "/cmnfs/proj/prosit_astral/datasets"

match config['dataloader']['dataset']:
    case 'small':
        train_data_source = "/cmnfs/data/proteomics/Prosit_PTMs/Transformer_Train/clean_train.parquet"
        val_data_source =   "/cmnfs/data/proteomics/Prosit_PTMs/Transformer_Train/clean_val.parquet"
        test_data_source =  "/cmnfs/data/proteomics/Prosit_PTMs/Transformer_Train/clean_test.parquet"
        steps_per_epoch = 7_992 / config['dataloader']['batch_size']
    case 'full':
        train_data_source = "/cmnfs/data/proteomics/Prosit_PTMs/Transformer_Train/no_aug_train.parquet"
        val_data_source =   "/cmnfs/data/proteomics/Prosit_PTMs/Transformer_Train/no_aug_val.parquet"
        test_data_source =  "/cmnfs/data/proteomics/Prosit_PTMs/Transformer_Train/no_aug_test.parquet"
        steps_per_epoch = 21_263_168 / config['dataloader']['batch_size']
    case 'combined':
        train_data_source = "/nfs/home/students/d.lochert/projects/astral/dlomix-astral/combined_dlomix_format_train.parquet"
        val_data_source =   "/nfs/home/students/d.lochert/projects/astral/dlomix-astral/combined_dlomix_format_val.parquet"
        test_data_source =  "/nfs/home/students/d.lochert/projects/astral/dlomix-astral/combined_dlomix_format_test.parquet"
        steps_per_epoch = 630_000 / config['dataloader']['batch_size']

# Faster loading if dataset is already saved
if config['dataloader']['load_data']:
   int_data = FragmentIonIntensityDataset.load_from_disk("/nfs/home/students/d.lochert/projects/astral/dlomix-astral/combined_dataset")
else:
    int_data = FragmentIonIntensityDataset(
        data_source=train_data_source,
        val_data_source=val_data_source,
        test_data_source=test_data_source,
        data_format="parquet", 
        # val_ratio=0.2, 
        max_seq_len=30, 
        encoding_scheme="naive-mods",
        alphabet=CUSTOM_ALPHABET,
        with_termini=False,
        model_features=["charge_oh", "collision_energy","method_nr_oh","machine_oh"],
        batch_size=config['dataloader']['batch_size']
    )



#int_data.save_to_disk("combined_dlomix.pt")

# print([m for m in int_data.tensor_train_data.take(1)][0][0])
# print([m for m in int_data.tensor_train_data.take(1)][0][1])





from models.models import TransformerModel
import wandb
from wandb.keras import WandbCallback

wandb.login(key='d6d86094362249082238642ed3a0380fde08761c')

sweep_config = {
    'method': 'bayes',  # You can use 'grid', 'random' or 'bayes' for Bayesian optimization
    'metric': {
        'name': 'val_loss',  # Which metric to optimize (from model training)
        'goal': 'minimize'   # Can be 'minimize' or 'maximize'
    },
    'parameters': {
        'lr_base': {
            'distribution': 'uniform',   # Uniform distribution for learning rate
            'min': 0.000025,             # Minimum value for lr_base
            'max': 0.001                 # Maximum value for lr_base
        },
    }
}


sweep_id = wandb.sweep(sweep_config, project="combined_data_lr_testing", entity="elizabeth-lochert-flx")

inp = [m for m in int_data.tensor_train_data.take(1)][0][0]




def train():
    # Initialize a new run for this sweep iteration
    wandb.init(entity='elizabeth-lochert-flx')

    # Access the sweep parameters via wandb.config
    config = wandb.config

    # Update the config with the sweep parameters
    train_settings['lr_base'] = config.lr_base

    # Re-create your model and optimizer with the updated parameters
    model = TransformerModel(**model_settings, seed=train_settings['seed'])
    optimizer = tf.keras.optimizers.Adam(learning_rate=config.lr_base)

    model.compile(optimizer=optimizer, 
                loss=masked_spectral_distance,
                metrics=[masked_pearson_correlation_distance])
    out = model(inp)


    # Fit the model and include the WandbCallback
    model.fit(
        int_data.tensor_train_data,
        validation_data=int_data.tensor_val_data,
        epochs=train_settings['epochs'],
        callbacks=[WandbCallback()]
    )

    # Evaluate the model and log results
    val_loss, val_accuracy = model.evaluate(int_data.tensor_val_data, verbose=0)
    wandb.log({'val_loss': val_loss, 'val_accuracy': val_accuracy})
    
    # Finish the run
    wandb.finish()

wandb.agent(sweep_id, function=train, count=10)  # `count` defines the number of runs

wandb.finish()
