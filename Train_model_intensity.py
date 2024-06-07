import os
os.environ['HF_HOME'] = "/cmnfs/proj/prosit_astral"
os.environ['HF_DATASETS_CACHE'] = "/cmnfs/proj/prosit_astral/datasets"

print("[UNIMOD:1]-K[UNIMOD:1]".count('[UNIMOD:' + '1' + ']'))

import numpy as np
import pandas as pd
import tensorflow as tf

from datasets import disable_caching
#disable_caching()

from dlomix.constants import PTMS_ALPHABET
from dlomix.losses import masked_spectral_distance, masked_pearson_correlation_distance
import yaml

# Check if conda and cuda is setup properly
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


################################################
#                  Dataset                     #
################################################

from dlomix.data import FragmentIonIntensityDataset
#from dlomix.data import load_processed_dataset

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

# Faster loading if dataset is already saved
#if os.path.exists(config['dataloader']['save_path'] + '/dataset_dict.json') and (config['dataloader']['dataset'] != 'small'):
#    int_data = load_processed_dataset(config['dataloader']['save_path'])
#else:
int_data = FragmentIonIntensityDataset(
    data_source=train_data_source,
    val_data_source=val_data_source,
    test_data_source=test_data_source,
    data_format="parquet", 
    val_ratio=0.2, 
    max_seq_len=30, 
    encoding_scheme="naive-mods",
    alphabet=PTMS_ALPHABET,
    model_features=["precursor_charge_onehot", "collision_energy_aligned_normed","method_nbr"],
    batch_size=config['dataloader']['batch_size']
)
#int_data.save_to_disk(config['dataloader']['save_path'])

#################################################
#         Choose and compile model              #
#################################################

optimizer = tf.keras.optimizers.Adam(learning_rate=train_settings['lr_base'])

if config['model_type'] == 'ours':
    from models.models import TransformerModel

    print("Loading Transformer Model")

    if model_settings['integration_method'] not in ['embed_input', 'pretoken', 'inject', 'adaptive']:
        raise ValueError("Invalid model setting for 'integration_method'")

    model = TransformerModel(**model_settings, seed=train_settings['seed'])

elif config['model_type'] == 'prosit_t':
    from models.prosit_t.models.prosit_transformer import PrositTransformerMean174M2 as PrositTransformer

    model = PrositTransformer(
        vocab_dict=PTMS_ALPHABET,
        **config['prosit']
    )

print("Compiling Transformer Model")
model.compile(optimizer=optimizer, 
            loss=masked_spectral_distance,
            metrics=[masked_pearson_correlation_distance])
inp = [m for m in int_data.tensor_train_data.take(1)][0][0]
out = model(inp)
model.summary()

print(len(int_data.tensor_train_data))

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
        project="astral" if config['wandb_settings']['project'] != None else config['wandb_settings']['project'],
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

class LLR(tf.keras.callbacks.Callback):
    def __init__(self, start_step=20000, end_step=50000, end_lr=1e-5):
        super(LLR, self).__init__()
        self.start_step = start_step
        self.end_step = end_step
        self.end_lr = end_lr
        self.initial_lr = None  # This will be set in the first on_train_batch_begin call

    def on_train_batch_begin(self, batch, logs=None):
        global_step = tf.keras.backend.get_value(self.model.optimizer.iterations)
        
        if self.initial_lr is None:
            self.initial_lr = tf.keras.backend.get_value(self.model.optimizer.learning_rate)
        
        if self.start_step <= global_step < self.end_step:
            progress = (global_step - self.start_step) / (self.end_step - self.start_step)
            new_lr = self.initial_lr - progress * (self.initial_lr - self.end_lr)
            tf.keras.backend.set_value(self.model.optimizer.learning_rate, new_lr)
        elif global_step >= self.end_step:
            tf.keras.backend.set_value(self.model.optimizer.learning_rate, self.end_lr)

class DecayLR(tf.keras.callbacks.Callback):
    # A decay learning rate that decreases the learning rate an order of magnitude
    # every mag_drop_every_n_steps steps.
    def __init__(self,
        mag_drop_every_n_steps=100000,      # 100000
        start_step=20000,                   # 20000
        end_step=1e10                       # 1e10
    ):
        super(DecayLR, self).__init__()
        self.alpha = np.exp(np.log(0.1) / mag_drop_every_n_steps)
        self.start_step = start_step
        self.end_step = end_step
        self.ticker = 1

    def on_train_batch_begin(self, batch, *args):
        #global_step = model.optimizer.variables[0].numpy()
        global_step = int(tf.keras.backend.get_value(self.model.optimizer.iterations))
        if (global_step >= self.start_step) & (global_step < self.end_step):
            current_lr = tf.keras.backend.get_value(self.model.optimizer.learning_rate)
            new_lr = current_lr * self.alpha
            tf.keras.backend.set_value(self.model.optimizer.learning_rate, new_lr)

class GeometricLR(tf.keras.callbacks.Callback):
    def __init__(self,
                 epoch_start,
                 lr_start,
                 decay_factor,
                 decay_constant
    ):
        super(GeometricLR, self).__init__()
        self.epoch_start = epoch_start
        self.lr_start = lr_start
        self.decay_factor = decay_factor
        self.decay_constant = decay_constant * steps_per_epoch

        self.step_start = epoch_start * steps_per_epoch

    def on_train_batch_begin(self, batch, *args):
        step = int(tf.keras.backend.get_value(self.model.optimizer.iterations))

        if step < self.step_start:
            lr_new = self.lr_start
        else:
            step_adj = step - self.step_start
            lr_new = self.lr_start * (self.decay_factor ** (step_adj / self.decay_constant))
        
        tf.keras.backend.set_value(self.model.optimizer.lr, lr_new)

class LinearLR(tf.keras.callbacks.Callback):
    def __init__(self,
                 epoch_start,
                 epoch_end,
                 lr_start,
                 lr_end,
    ):
        super(LinearLR, self).__init__()
        self.epoch_start = epoch_start
        self.epoch_end = epoch_end
        self.lr_start = lr_start
        self.lr_end = lr_end

        self.step_start = epoch_start * steps_per_epoch
        self.step_end = epoch_end * steps_per_epoch

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

if train_settings['lr_method'] == 'geometric':
    callbacks.append(GeometricLR(*train_settings['lr_geometric']))

elif train_settings['lr_method'] == 'linear':
    callbacks.append(LinearLR(*train_settings['lr_linear']))

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
