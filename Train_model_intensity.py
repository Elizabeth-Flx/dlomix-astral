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


with open("/nfs/home/students/d.lochert/projects/astral/dlomix/my_scripts/config.yaml", 'r') as yaml_file:
    config = yaml.safe_load(yaml_file)

model_settings = config['model_settings']
train_settings = config['train_settings']

print("DataLoader Settings:")
print(f"Dataset: {config['dataloader']['dataset']}")
print(f"Batch Size: {config['dataloader']['batch_size']}")

print("\nModel config:")
for key, value in model_settings.items():
    print(f"{key}: {value}")

print("\nTraining Settings:")
for key, value in train_settings.items():
    print(f"{key}: {value}")
print('='*32)


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

#print(type(rt_data.tensor_train_data))
#print(type(rt_data.tensor_val_data))

import wandb
from wandb.keras import WandbCallback

wandb.login(key='d6d86094362249082238642ed3a0380fde08761c')
#wandb.init(project='astral', entity='elizabeth-lochert-flx')

#print(rt_data.dataset)

from dlomix.models import PrositIntensityPredictor
from dlomix.constants import PTMS_ALPHABET
from dlomix.losses import masked_spectral_distance, masked_pearson_correlation_distance
import tensorflow as tf

optimizer = tf.keras.optimizers.Adam(learning_rate=train_settings['lr_base'])

#model = PrositIntensityPredictor(vocab_dict=PTMS_ALPHABET)

from models.models import TransformerModel

print("Loading Transformer Model")

if model_settings['prec_type'] not in ['embed_input', 'pretoken', 'inject']:
    raise ValueError("Invalid model setting for 'prec_type'")

model = TransformerModel(**model_settings)

print("Compiling Transformer Model")
model.compile(optimizer=optimizer, 
            loss=masked_spectral_distance,
            metrics=[masked_pearson_correlation_distance])


# Callbacks

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from callbacks import CyclicLR, LearningRateLogging

cyclicLR = CyclicLR(
    base_lr=train_settings['lr_base'],
    max_lr=train_settings['lr_max'],
    step_size=2,
    mode='triangular',
    gamma=0.95
)

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



# Wandb init

import random  
from string import ascii_lowercase, ascii_uppercase, digits
chars = ascii_lowercase + ascii_uppercase + digits

#name =  config['dataloader']['dataset'][0] + '_' + \
#        model_settings['prec_type'] + '_' + \
#        (str(model_settings['inject_pre'])[0] +  
#        str(model_settings['inject_post'])[0] +
#        model_settings['inject_position'] + '_') \
#            if model_settings['prec_type']=='inject' else "" + \
#        'd' + str(model_settings['depth']) + '_' + \
#        train_settings['lr_method'] + '_' + \
#        ''.join([random.choice(chars) for _ in range(3)])

name = f"%s_%s%s_d%s_%s_%s_%s" % ( 
    config['dataloader']['dataset'][0],
    model_settings['prec_type'],
    (str(model_settings['inject_pre'])[0] +  str(model_settings['inject_post'])[0] + model_settings['inject_position'])
        if model_settings['prec_type']=='inject' else "",
    model_settings['depth'],
    train_settings['lr_method'],
    train_settings['lr_base'],
    ''.join([random.choice(chars) for _ in range(3)])
)


tags = [
    config['dataloader']['dataset'],
    'depth_' + str(model_settings['depth']),
    'prec_type_' + model_settings['prec_type'],
    'lr_method_' + train_settings['lr_method'],
    'lr_base_' + str(train_settings['lr_base']),
    'lr_max_' + str(train_settings['lr_max']),
]
tags + [model_settings['inject_pre'], 
        model_settings['inject_post'], 
        model_settings['inject_position']] if model_settings['prec_type'] == 'inject' else []

wandb.init(
    project="lr_testing",
    name=name,
    tags=tags,
    config=config,
    entity='elizabeth-lochert-flx'
)

callbacks = [
    WandbCallback(save_model=False),
    #early_stopping,
    learningRate
]

if train_settings['lr_method'] == 'cyclic':
    callbacks.append(cyclicLR)

model.fit(
    rt_data.tensor_train_data,
    validation_data=rt_data.tensor_val_data,
    epochs=train_settings['epochs'],
    callbacks=callbacks
)

print(model.summary())

wandb.finish()

#model.save('Prosit_cit/Intensity/')
