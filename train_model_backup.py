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


with open("/nfs/home/students/d.lochert/projects/astral/dlomix/my_scripts/config.yaml", 'r') as yaml_file:
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

full_data = pd.read_parquet('/nfs/home/students/d.lochert/projects/astral/dlomix-astral/combined_broken_collision_energy.parquet')

# Shuffle data
df = full_data.sample(frac=1, random_state=42)

def map_protein_sequence(seq):
    return np.array([protein_map[aa] for aa in seq] + [0] * (MAX_SEQUENCE_LENGTH - len(seq)))

df['encoded_seqeunce'] = df['prosit_sequence'].apply(map_protein_sequence)
df['precursor_charge_onehot'] = df['charge'].apply(lambda x: np.array([1 if i == x else 0 for i in range(1, 7)]))

nrows = len(df)

print(nrows)

# Split data into train, val and test
train_data = df.iloc[               :int(nrows*0.8)]
val_data   = df.iloc[int(nrows*0.8) :int(nrows*0.9)]
test_data  = df.iloc[int(nrows*0.9) :]

print("train", train_data.shape)
print("val  ", val_data.shape)
print("test ", test_data.shape)

# transform dataframes to dictionaries
train_data = train_data.to_dict('records')
val_data   = val_data  .to_dict('records')
test_data  = test_data .to_dict('records')

# print("train", train_data[0])
# print("val  ", val_data[0])
# print("test ", test_data[0])



from models.models import TransformerModel

print("Loading Transformer Model")

model = TransformerModel(**model_settings, seed=train_settings['seed'])

optimizer = tf.keras.optimizers.Adam(learning_rate=train_settings['lr_base'])

print("Compiling Transformer Model")
model.compile(optimizer=optimizer, 
            loss=masked_spectral_distance,
            metrics=[masked_pearson_correlation_distance])
#inp = [m for m in int_data.tensor_train_data.take(1)][0][0]
out = model(train_data[0])
model.summary()


# stop code
raise Exception('Stop code') 

# print(len(int_data.tensor_train_data))

###################################################
#                   Wandb init                    #
###################################################

import wandb
WandbCallback  = wandb.keras.WandbCallback
from wandb.keras import WandbCallback

import random  
from string import ascii_lowercase, ascii_uppercase, digits
chars = ascii_lowercase + ascii_uppercase + digits

name = f"%s_%s%s_d%s_%s_%s_%s" % ( 
    config['dataloader']['dataset'][0],
    model_settings['integration_method'],
    (str(model_settings['inject_pre'])[0] +  str(model_settings['inject_post'])[0] + model_settings['inject_position'])
        if model_settings['integration_method']=='inject' else "",
    model_settings['depth'],
    train_settings['lr_method'],
    train_settings['lr_base'],
    ''.join([random.choice(chars) for _ in range(3)])
)

tags = [
    config['dataloader']['dataset'],
    'depth_' + str(model_settings['depth']),
    'int_method_' + model_settings['integration_method'],
    'lr_method_' + train_settings['lr_method'],
    'lr_base_' + str(train_settings['lr_base']),
]
tags + [model_settings['inject_pre'], 
        model_settings['inject_post'], 
        model_settings['inject_position']] if model_settings['integration_method'] == 'inject' else []

if train_settings['log_wandb']:
    wandb.login(key='d6d86094362249082238642ed3a0380fde08761c')
    wandb.init(
        project="astral" if config['wandb_settings']['project'] == None else config['wandb_settings']['project'],
        name=name,
        tags=tags,
        config=config,
        entity='elizabeth-lochert-flx'
    )

#######################################################
#                      Callbacks                      #
#######################################################

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
#from callbacks import CyclicLR, LearningRateLogging

early_stopping = EarlyStopping(
    monitor="val_loss",
    min_delta=0.001,
    patience=20,
    restore_best_weights=True)

# ValueError: When using `save_weights_only=True` in `ModelCheckpoint`, the filepath provided must end in `.weights.h5` (Keras weights format). Received: filepath=saved_models/best_model_intensity_nan.keras
save_best = ModelCheckpoint(
    'saved_models/best_model_intensity_nan.keras',
    monitor="val_loss",
    verbose=1,
    save_best_only=True,
    save_weights_only=True
)

#cyclicLR = CyclicLR(
#    base_lr=train_settings['lr_base'],
#    max_lr=train_settings['lr_max'],
#    step_size=2,
#    mode='triangular',
#    gamma=0.95
#)

class CyclicLR(tf.keras.callbacks.Callback):
    pass

# class GeometricLR(tf.keras.callbacks.Callback):
#     def __init__(self,
#                  epoch_start,
#                  lr_start,
#                  decay_factor,
#                  decay_constant
#     ):
#         super(GeometricLR, self).__init__()
#         self.epoch_start = epoch_start
#         self.lr_start = lr_start
#         self.decay_factor = decay_factor
#         self.decay_constant = decay_constant * steps_per_epoch

#         self.step_start = epoch_start * steps_per_epoch

    def on_train_batch_begin(self, batch, *args):
        step = int(tf.keras.backend.get_value(self.model.optimizer.iterations))

        if step < self.step_start:
            lr_new = self.lr_start
        else:
            step_adj = step - self.step_start
            lr_new = self.lr_start * (self.decay_factor ** (step_adj / self.decay_constant))
        
        tf.keras.backend.set_value(self.model.optimizer.lr, lr_new)

# class LinearLR(tf.keras.callbacks.Callback):
#     def __init__(self,
#                  epoch_start,
#                  epoch_end,
#                  lr_start,
#                  lr_end,
#     ):
#         super(LinearLR, self).__init__()
#         self.epoch_start = epoch_start
#         self.epoch_end = epoch_end
#         self.lr_start = lr_start
#         self.lr_end = lr_end

#         self.step_start = epoch_start * steps_per_epoch
#         self.step_end = epoch_end * steps_per_epoch

    def on_train_batch_begin(self, batch, *args):
        step = int(tf.keras.backend.get_value(self.model.optimizer.iterations))

        if step < self.step_start:
            lr_new = self.lr_start
        elif step > self.step_end:
            lr_new = self.lr_end
        else:
            step_adj = step - self.step_start
            lr_new = self.lr_start + step_adj * (self.lr_end - self.lr_start) / (self.step_end - self.step_start)
        
        tf.keras.backend.set_value(self.model.optimizer.lr, lr_new)



class LearningRateReporter(tf.keras.callbacks.Callback):
    def on_train_batch_end(self, batch, *args):
        wandb.log({'learning_rate': self.model.optimizer._learning_rate.numpy()})
    
callbacks = []
if train_settings['log_wandb']:
    callbacks.append(WandbCallback(save_model=False))
    callbacks.append(LearningRateReporter())

# if train_settings['lr_method'] == 'geometric':
#     callbacks.append(GeometricLR(*train_settings['lr_geometric']))

# elif train_settings['lr_method'] == 'linear':
#     callbacks.append(LinearLR(*train_settings['lr_linear']))

#elif train_settings['lr_method'] == 'decay':
#    for lr in train_settings['lr_decay']:
#        callbacks.append(DecayLR(*lr))

##############################################################
#                       Train model                          #
##############################################################

model.fit(
    int_data.tensor_train_data,
    validation_data=int_data.tensor_val_data,
    epochs=train_settings['epochs'],
    callbacks=callbacks
)

if train_settings['log_wandb']:
    wandb.finish()

# Evaluate model on training data
train_loss, train_accuracy = model.evaluate(int_data.tensor_train_data, verbose=0)
print(f"Training Loss: {train_loss:.4f}")

# Evaluate model on validation data
val_loss, val_accuracy = model.evaluate(int_data.tensor_val_data, verbose=0)
print(f"Validation Loss: {val_loss:.4f}")

#model.save('Prosit_cit/Intensity/')