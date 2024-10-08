import tensorflow as tf
K = tf.keras
L = K.layers
I = K.initializers

import models.model_parts as mp
import numpy as np

ALPHABET_UNMOD = {
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

class TransformerModel(K.Model):
    def __init__(self,
        running_units=256,
        d=64,
        h=4,
        ffn_mult=1,
        depth=3,
        pos_type='learned', # learned
        integration_method="embed_input", # embed_input | single_token | multi_token | inject
        # token_combine ='add',    # add | mult
        learned_pos=False,
        prenorm=True,
        norm_type="layer",      # layer | batch | adaptive
        penultimate_units=None,
        output_units=174,
        # max_charge=6,
        sequence_length=30,
        alphabet=False,
        dropout=0,
        precursor_units=None,
        inject_pre=True,        # inject before Attention block
        inject_post=True,       # inject into FNN
        inject_position="all",  # all | first | last
        seed=42,
        identiy_metadata=False,
    ):
        tf.random.set_seed(seed)

        super(TransformerModel, self).__init__()
        self.ru = running_units
        self.depth = depth
        self.integration_method = integration_method
        self.prec_units = running_units if precursor_units == None else precursor_units

        self.inject_pre = inject_pre 
        self.inject_post = inject_post
        self.inject_position = inject_position


        # Positional Encoding 
        if learned_pos:
            self.pos = tf.Variable(tf.random.normal((sequence_length, running_units)), trainable=True)
        else:
            self.pos = tf.Variable(mp.FourierFeatures(tf.range(1000, dtype=tf.float32), 1, 150, running_units), trainable=False)
        self.alpha_pos = tf.Variable(0.1, trainable=True)
        
        # Beginning
        #self.string_lookup = preprocessing.StringLookup(vocabulary=list(ALPHABET_UNMOD.keys()))
        #self.string_lookup.build(None, 30)
        
        #self.embedding = L.Embedding(len(ALPHABET_UNMOD), running_units, input_length=sequence_length)
        self.first = L.Dense(running_units)

        penultimate_units = running_units if penultimate_units is None else penultimate_units


        if identiy_metadata:
            self.mult_factor = 1
            self.meta_weight_init = I.Zeros
        else:
            self.mult_factor = 0
            self.meta_weight_init = I.GlorotNormal
        

        # Metadata integration
        if integration_method == 'multi_token':
            self.char_embedder = L.Dense(running_units, kernel_initializer=self.meta_weight_init()) # this should be changed to its own variable
            self.ener_embedder = L.Dense(running_units, kernel_initializer=self.meta_weight_init()) # this should be changed to its own variable (possibly change to fourier feature (PrecursorToken))
            self.meth_embedder = L.Dense(running_units, kernel_initializer=self.meta_weight_init()) # this should be changed to its own variable
            self.mach_embedder = L.Dense(running_units, kernel_initializer=self.meta_weight_init()) # this should be changed to its own variable
        else:
            self.meta_dense = L.Dense(running_units, kernel_initializer=self.meta_weight_init())
        
            if integration_method in ['single_token', 'token_sum', 'token_mult', 'single_sum', 'single_mult']:
                self.metadata_encoder = L.Dense(running_units, kernel_initializer=self.meta_weight_init())
            elif integration_method == 'FiLM_full':
                self.metadata_encoder = L.Dense(running_units*2*depth, kernel_initializer=self.meta_weight_init())
            elif integration_method == 'FiLM_reduced':
                self.metadata_encoder = L.Dense(2*depth, kernel_initializer=self.meta_weight_init())    # with alpha and beta (if only beta just depth)
            elif integration_method in ['penult_sum', 'penult_mult']:
                self.metadata_encoder = L.Dense(penultimate_units, kernel_initializer=self.meta_weight_init())
            elif integration_method == 'embed_input': # todo add parameter to config that can choose if none given use ru
                self.metadata_encoder = L.Dense(running_units, kernel_initializer=self.meta_weight_init())
            elif integration_method in ['FiLM_sum', 'FiLM_mult']:
                self.metadata_encoder = L.Dense(running_units*depth, kernel_initializer=self.meta_weight_init())
            elif integration_method == 'single_both':
                self.metadata_encoder = L.Dense(running_units*2, kernel_initializer=self.meta_weight_init())

        # Middle
        attention_dict = {
            'd': d,
            'h': h,
            'dropout': dropout,
            'alphabet': alphabet,
        }
        ffn_dict = {
            'unit_multiplier': ffn_mult,
            'dropout': dropout,
            'alphabet': alphabet,
        }
        self.main = [ # todo remove adaptive norm idea and iject
            mp.TransBlock(
                attention_dict, 
                ffn_dict, 
                prenorm=prenorm, 
                norm_type=norm_type,    # layer | batch | adaptive
                use_embed=True if   (integration_method=='inject') and
                                    (inject_position == 'all' or
                                     inject_position == 'first' and i == 0 or
                                     inject_position == 'last' and i == depth-1) else False,     # Creates self.embed in model_parts which is used to integrate metadata into model
                preembed=inject_pre,
                postembed=inject_post,
                is_cross=False,
                seed=seed
            )
            for i in range(depth)
        ]

        # End
        self.penultimate_dense = L.Dense(penultimate_units)
        self.penultimate_norm = K.Sequential([
            L.LayerNormalization(),
            L.ReLU()
        ])
        
        self.final = L.Dense(output_units, activation='sigmoid')

    
    def MetadataGenerator(self, char_oh, ener, meth_oh, mach_oh): # todo rename

        # method multi_token requires each attribute to have their own embedding
        if self.integration_method == 'multi_token':
            char_token = self.char_embedder(char_oh)        [:, None]   # (bs, 1, ru)
            ener_token = self.ener_embedder(ener[:, None])  [:, None]   # (bs, 1, ru)
            meth_token = self.meth_embedder(meth_oh)        [:, None]   # (bs, 1, ru)
            mach_token = self.mach_embedder(mach_oh)        [:, None]   # (bs, 1, ru)

            return tf.concat([char_token, ener_token, meth_token, mach_token], axis=1)  # (bs, 4, ru)

        # all other methods require consolidated outputs (generated by MetaEmbedder)
        combined_meta = tf.concat([char_oh, meth_oh, mach_oh, ener[:, None]], axis=1) # (bs, 6+2+3+1)

        # ablation study
        combined_meta = tf.concat([char_oh], axis=1) # (bs, 6+2+3+1)

        # print(combined_meta)
        # print(combined_meta.shape)

        # control study with no metadata
        # if combined_meta.shape[0] == None:
        #     combined_meta = tf.zeros((1, 12))
        # else:
        #     combined_meta = tf.zeros((combined_meta.shape[0], 12))

        return self.MetaEnocoder(combined_meta)


    # If we want to add more layers in the metadata encoding do it here
    def MetaEnocoder(self, x): # todo rename/merge with MetadataGenerator
        x = self.meta_dense(x)
        return self.metadata_encoder(x)[:, None]



    def Main(self, x, metadata=None):     # todo alter to work with integration methods
        out = x

        if self.integration_method in ['FiLM_full', 'FiLM_reduced', 'FiLM_sum', 'FiLM_mult']:
            metadata = tf.split(metadata, self.depth, axis=-1)

        for i in range(len(self.main)):
            layer = self.main[i]

            if self.integration_method in ['FiLM_full', 'FiLM_reduced']:
                gamma, beta = tf.split(metadata[i], 2, axis=-1)
                out = out * (gamma + self.mult_factor) + beta
            elif self.integration_method == 'token_sum' or (i==0 and self.integration_method=='single_sum'):    # these should have a gate variable alpha 
                out = out + metadata
            elif self.integration_method == 'token_mult' or (i==0 and self.integration_method=='single_mult'): 
                out = out * (metadata + self.mult_factor)
            elif self.integration_method == 'FiLM_sum':
                out = out + metadata[i]
            elif self.integration_method == 'FiLM_mult':
                out = out * (metadata[i] + self.mult_factor)
            elif self.integration_method == 'single_both' and i==0:
                alpha, beta = tf.split(metadata, 2, axis=-1)
                out = out * (alpha + self.mult_factor) + beta

            out = layer(out, None) # todo maybe implement inject? (low prio)


            # if (self.inject_position == "all") or \
            #    (self.inject_position == "first" and i == 0) or \
            #    (self.inject_position == "last" and i == len(self.main) - 1):
            #     out = layer(out, temb=tb_emb)
            # else:
            #     out = layer(out, None)

        return out

    def call(self, x, training=False):

        sequence = x['modified_sequence']
        char_oh = x['charge_oh']
        ener    = x['collision_energy']
        meth_oh = x['method_nr_oh']
        mach_oh = x['machine_oh']

        # print(char_oh.shape)
        # print(ener   .shape)
        # print(meth_oh.shape)
        # print(mach_oh.shape)

        # onehot encode sequence
        out = tf.one_hot(tf.cast(sequence, tf.int32), len(ALPHABET_UNMOD))

        metadata = self.MetadataGenerator(char_oh, ener, meth_oh, mach_oh)

        out = self.first(out) + self.alpha_pos*self.pos[:out.shape[1]]  # todo check about this positional encoding (seems wierd)

        if self.integration_method in ['single_token', 'multi_token']:           # todo make this better (low prio)
            out = tf.concat([out, metadata], axis=1)

        out = self.Main(out, metadata)     # Transformer blocks

        # Penultimate 
        out = self.penultimate_dense(out)

        if self.integration_method == 'penult_sum':
            out = out + metadata
        elif self.integration_method == 'penult_mult':
            out = out * metadata

        out = self.penultimate_norm(out)

        # Final
        out = self.final(out)
        return tf.reduce_mean(out, axis=1)
    

    def get_meta_vector(self, x):
        sequence = x['modified_sequence']
        char_oh = x['charge_oh']
        ener    = x['collision_energy']
        meth_oh = x['method_nr_oh']
        mach_oh = x['machine_oh']

        # onehot encode sequence
        out = tf.one_hot(tf.cast(sequence, tf.int32), len(ALPHABET_UNMOD))

        metadata = self.MetadataGenerator(char_oh, ener, meth_oh, mach_oh)

        return metadata     # (bs, variable, ru/penult_units)


