import os
os.environ['HF_HOME'] = "/cmnfs/proj/prosit_astral"
os.environ['HF_DATASETS_CACHE'] = "/cmnfs/proj/prosit_astral/datasets"

print("[UNIMOD:1]-K[UNIMOD:1]".count('[UNIMOD:' + '1' + ']'))

import numpy as np
from dlomix.data import FragmentIonIntensityDataset
import pandas as pd

from datasets import disable_caching
disable_caching()

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

rt_data = FragmentIonIntensityDataset(
    data_source="/cmnfs/data/proteomics/Prosit_PTMs/Transformer_Train/clean_train.parquet",
    val_data_source="/cmnfs/data/proteomics/Prosit_PTMs/Transformer_Train/clean_val.parquet",
    test_data_source="/cmnfs/data/proteomics/Prosit_PTMs/Transformer_Train/clean_test.parquet",
    data_format="parquet", 
    val_ratio=0.2, max_seq_len=30, encoding_scheme="naive-mods",
    vocab=PTMS_ALPHABET,
    model_features=["precursor_charge_onehot", "collision_energy_aligned_normed","method_nbr"],
    batch_size=2048
)

print(type(rt_data.tensor_train_data))
print(type(rt_data.tensor_val_data))

#import wandb
#from wandb.keras import WandbCallback
#
#wandb.login(key='d6d86094362249082238642ed3a0380fde08761c')
#wandb.init(project='astral', entity='elizabeth-lochert-flx')

#print(rt_data.dataset)

from dlomix.models import PrositIntensityPredictor
from dlomix.constants import PTMS_ALPHABET
from dlomix.losses import masked_spectral_distance, masked_pearson_correlation_distance
import tensorflow as tf

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

#model = PrositIntensityPredictor(vocab_dict=PTMS_ALPHABET)

from models.models import TransformerModel

print("Loading Transformer Model")

model = TransformerModel(
    running_units=128, 
    d=16,
    depth=3,
    ffn_mult=1, 
    penultimate_units=512,
    alphabet=False,
    dropout=0.1,
    prec_type='inject_pre',
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
    epochs=100,
    callbacks=[
#        WandbCallback(save_model=False),
        cyclicLR,
        early_stopping,
        #save_best,
        learningRate
    ]
)

#wandb.finish()

#model.save('Prosit_cit/Intensity/')
