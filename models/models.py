import tensorflow as tf
K = tf.keras
L = K.layers
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
        token_combine ='add',    # add | mult
        learned_pos=False,
        prenorm=True,
        norm_type="layer",      # layer | batch | adaptive
        penultimate_units=None,
        output_units=174,
        max_charge=6,
        sequence_length=30,
        alphabet=False,
        dropout=0,
        precursor_units=None,
        inject_pre=True,        # inject before Attention block
        inject_post=True,       # inject into FNN
        inject_position="all",  # all | first | last
        seed=42
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

        self.token_combine = token_combine


        # Positional
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
        if integration_method in ['multi_token', 'single_token', 'token_sum', 'inject', 'adaptive', 'token_mult', 'FiLM_small', 'FiLM_large', 'penult_add']:
            self.charge_embedder = L.Dense(running_units) #mp.PrecursorToken(running_units, 64, 1, 15)
            self.ce_embedder = mp.PrecursorToken(running_units, running_units, 0.01, 1.5)

        if integration_method == 'FiLM_small':
            self.film_gamma = tf.Variable(tf.ones((1,1,running_units)), name="film_gamma")
            self.film_beta = tf.Variable(tf.zeros((1,1,running_units)), name="film_beta")

        if integration_method == 'FiLM_large':
            self.film_gamma = tf.Variable(tf.ones((1,sequence_length,running_units)), name="film_gamma")
            self.film_beta = tf.Variable(tf.zeros((1,sequence_length,running_units)), name="film_beta")

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
        self.main = [
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
        penultimate_units = running_units if penultimate_units is None else penultimate_units

        if integration_method == 'penult_add':
            self.meta_encoder = L.Dense(penultimate_units)
            self.prepenult = L.Dense(penultimate_units)
        
        self.penultimate = K.Sequential([
            *( [L.Dense(penultimate_units)] if integration_method != 'penult_add' else [] ),
            L.LayerNormalization(),
            L.ReLU()
        ])
        
        self.final = L.Dense(output_units, activation='sigmoid')

    def EmbedInputs(self, sequence, precursor_charge, collision_energy):
        #print(sequence)
        length = sequence.shape[1]
        #input_embedding = tf.one_hot(self.string_lookup(sequence), len(ALPHABET_UNMOD))
        input_embedding = tf.one_hot(tf.cast(sequence, tf.int32), len(ALPHABET_UNMOD))
        if self.integration_method == 'embed_input':
            #print(precursor_charge.shape)
            #print(precursor_charge[:,None].shape)
            charge_emb = tf.tile(precursor_charge[:,None], [1, length, 1])          # (bs, 1, 6)
            #print(charge_emb.shape)
            #print(collision_energy.shape)
            #print(collision_energy[:,None][:,None].shape)
            ce_emb = tf.tile(collision_energy[:,None][:,None], [1, length, 1])      # (bs, 1, 1)
            #print(ce_emb.shape)

            input_embedding = tf.concat([input_embedding, tf.cast(charge_emb, tf.float32), ce_emb], axis=-1)
        
        return input_embedding

    def Main(self, x, tb_emb=None):
        out = x
        for i in range(len(self.main)):
            layer = self.main[i]

            if (self.inject_position == "all") or \
               (self.inject_position == "first" and i == 0) or \
               (self.inject_position == "last" and i == len(self.main) - 1):
                out = layer(out, temb=tb_emb)
            else:
                out = layer(out, None)

        return out

    def call(self, x, training=False):

        sequence = x['encoded_seqeunce']
        precchar = x['precursor_charge_onehot']
        collener = x['collision_energy']

        out = self.EmbedInputs(sequence, precchar, collener)

        out = self.first(out) + self.alpha_pos*self.pos[:out.shape[1]]
        tb_emb = None

        # === Methods that alter the transformer tokens === #
        if self.integration_method in ['multi_token', 'single_token', 'token_sum', 'token_mult', 'FiLM_small', 'FiLM_large', 
                                       'inject', 'adaptive', 'penult_add', 'penult_mult']:
            charge_token = self.charge_embedder(precchar)    # (bs, ru)
            ce_token = self.ce_embedder(collener)            # (bs, ru)

            if self.token_combine == 'add':
                combined_token = (charge_token + ce_token)[:,None]      # (bs, 1, ru)
            elif self.token_combine == 'mult':
                combined_token = (charge_token * ce_token)[:,None]      # (bs, 1, ru)

        match self.integration_method:
            case 'multi_token':
                out = tf.concat([charge_token[:,None], ce_token[:,None], out], axis=1)
            case 'single_token':
                out = tf.concat([combined_token, out], axis=1)
            case 'token_sum':
                out = out + combined_token
            case 'token_mult':
                out = out * combined_token
            case 'FiLM_small' | 'FiLM_large':
                out = out * (self.film_gamma * combined_token) + self.film_beta * combined_token
            case 'inject' | 'adaptive' | 'penult_add' | 'penult_mult':
                tb_emb = tf.concat([charge_token, ce_token], axis=-1)   # (bs, 2*running_units)

        out = self.Main(out, tb_emb=tb_emb)     # Transformer blocks

        if self.integration_method == 'penult_add':
            out = self.prepenult(out) + self.meta_encoder(tb_emb)[:,None]
        elif self.integration_method == 'penult_mult':
            out = self.prepenult(out) * self.meta_encoder(tb_emb)[:,None]

        out = self.penultimate(out)

        out = self.final(out)

        return tf.reduce_mean(out, axis=1)

