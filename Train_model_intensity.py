import os
os.environ['HF_HOME'] = "/cmnfs/proj/prosit_astral"
os.environ['HF_DATASETS_CACHE'] = "/cmnfs/proj/prosit_astral/datasets"

print("[UNIMOD:1]-K[UNIMOD:1]".count('[UNIMOD:' + '1' + ']'))

import numpy as np
from dlomix.data import FragmentIonIntensityDataset
import pandas as pd

from datasets import disable_caching
#disable_caching()

import tensorflow as tf
print('='*32)
print('Conda info')
print(f"Environment: {os.environ['CONDA_DEFAULT_ENV']}")
print('='*32)
print('Tensorflow info')
print(f"Version: {tf.__version__}")
print(f"Built with CUDA: {tf.test.is_built_with_cuda()}")
print(f"Number of GPUs available: {len(tf.config.list_physical_devices('GPU'))}")
print(f"List of GPUs available: {tf.config.list_physical_devices('GPU')}")
print('='*32)



PTMS_ALPHABET = {
    "A": 1,
    "C": 2,
    "D": 3,
    "E": 4,
    "F": 5,
    "G": 6,
    "H": 7,
    "I": 8,
    "K": 9,
    "L": 10,
    "M": 11,
    "N": 12,
    "P": 13,
    "Q": 14,
    "R": 15,
    "S": 16,
    "T": 17,
    "V": 18,
    "W": 19,
    "Y": 20,
    "M[UNIMOD:35]": 21,
    "R[UNIMOD:7]":22,
    "C[UNIMOD:4]": 2,
    "Q[UNIMOD:7]":4,
    "N[UNIMOD:7]":3,
}

import yaml


with open("./config.yaml", 'r') as yaml_file:
    config = yaml.safe_load(yaml_file)

print("DataLoader Settings:")
print(f"Dataset: {config['dataloader']['dataset']}")
print(f"Batch Size: {config['dataloader']['batch_size']}")

print("\nModel config:")
print(f"Running Units: {config['model_settings']['running_units']}")
print(f"d: {config['model_settings']['d']}")
print(f"Depth: {config['model_settings']['depth']}")
print(f"FFN Multiplier: {config['model_settings']['ffn_mult']}")
print(f"Penultimate Units: {config['model_settings']['penultimate_units']}")
print(f"Alphabet: {config['model_settings']['alphabet']}")
print(f"Dropout: {config['model_settings']['dropout']}")
print(f"Prec Type: {config['model_settings']['prec_type']}")
print(f"Inject Position: {config['model_settings']['inject_position']}")

print("\nTraining Settings:")
print(f"Epochs: {config['train_settings']['epochs']}")


match config['dataloader']['dataset']:
    case 'small':
        train_data_source = "/cmnfs/data/proteomics/Prosit_PTMs/Transformer_Train/clean_train.parquet"
        val_data_source =   "/cmnfs/data/proteomics/Prosit_PTMs/Transformer_Train/clean_val.parquet"
        test_data_source =  "/cmnfs/data/proteomics/Prosit_PTMs/Transformer_Train/clean_test.parquet"
    case 'full':
        train_data_source = "/cmnfs/data/proteomics/Prosit_PTMs/Transformer_Train/no_aug_train.parquet"
        val_data_source =   "/cmnfs/data/proteomics/Prosit_PTMs/Transformer_Train/no_aug_val.parquet"
        test_data_source =  "/cmnfs/data/proteomics/Prosit_PTMs/Transformer_Train/no_aug_test.parquet"


rt_data = FragmentIonIntensityDataset(
    data_source=train_data_source,
    val_data_source=val_data_source,
    test_data_source=test_data_source,
    data_format="parquet", 
    val_ratio=0.2, max_seq_len=30, encoding_scheme="naive-mods",
    vocab=PTMS_ALPHABET,
    model_features=["precursor_charge_onehot", "collision_energy_aligned_normed","method_nbr"],
    batch_size=config['dataloader']['batch_size']
)

print(type(rt_data.tensor_train_data))
print(type(rt_data.tensor_val_data))

import wandb
from wandb.keras import WandbCallback

wandb.login(key='d6d86094362249082238642ed3a0380fde08761c')
wandb.init(project='astral', entity='elizabeth-lochert-flx')

#print(rt_data.dataset)

from dlomix.models import PrositIntensityPredictor
from dlomix.constants import PTMS_ALPHABET
from dlomix.losses import masked_spectral_distance, masked_pearson_correlation_distance
import tensorflow as tf

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

#model = PrositIntensityPredictor(vocab_dict=PTMS_ALPHABET)

from models.models import TransformerModel

print("Loading Transformer Model")

model_settings = config['model_settings']

model = TransformerModel(
    running_units=model_settings['running_units'], 
    d=model_settings['d'],
    depth=model_settings['depth'],
    ffn_mult=model_settings['ffn_mult'], 
    penultimate_units=model_settings['penultimate_units'],
    alphabet=False,
    dropout=0.1,
    prec_type=model_settings['prec_type'],              # embed_input | pretoken | inject_pre | inject_ffn
    inject_position=model_settings['inject_position']   # all | pre | post (only for inject_pre and inject_ffn)
)

print("Compiling Transformer Model")
model.compile(optimizer='adam', 
            loss=masked_spectral_distance,
            metrics=[masked_pearson_correlation_distance])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from callbacks import CyclicLR, LearningRateLogging

cyclicLR = CyclicLR(base_lr=0.000001, max_lr=0.0002, step_size=2, mode='triangular',
                 gamma=0.95)

early_stopping = EarlyStopping(
    monitor="val_loss",
    min_delta=0.001,
    patience=20,
    restore_best_weights=True)

# ValueError: When using `save_weights_only=True` in `ModelCheckpoint`, the filepath provided must end in `.weights.h5` (Keras weights format). Received: filepath=saved_models/best_model_intensity_nan.keras
"""save_best = ModelCheckpoint(
    'saved_models/best_model_intensity_nan.keras',
    monitor="val_loss",
    verbose=1,
    save_best_only=True,
    save_weights_only=True
)"""

learningRate = LearningRateLogging()

model.fit(
    rt_data.tensor_train_data,
    validation_data=rt_data.tensor_val_data,
    epochs=config['train_settings']['epochs'],
    callbacks=[
        WandbCallback(save_model=False),
        #cyclicLR,
        #early_stopping,
        #save_best,
        #learningRate
    ]
)

print(model.summary())

wandb.finish()

#model.save('Prosit_cit/Intensity/')
