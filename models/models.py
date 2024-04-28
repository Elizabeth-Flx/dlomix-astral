import tensorflow as tf
K = tf.keras
L = K.layers
import models.model_parts as mp
from tensorflow.keras.layers.experimental import preprocessing

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
        prec_type="embed_input", # embed_input | pretoken | inject_pre | inject_ffn
        learned_pos=True,
        prenorm=False,
        norm_type="layer",
        penultimate_units=None,
        output_units=174,
        max_charge=6,
        sequence_length=30,
        alphabet=False,
        dropout=0,
        precursor_units=None,
        inject_position="all" # all | pre | post
    ):
        super(TransformerModel, self).__init__()
        self.ru = running_units
        self.depth = depth
        self.prec_type = prec_type
        self.prec_units = running_units if precursor_units == None else precursor_units
        self.inject_pos = inject_position
        
        # Positional
        if learned_pos:
            self.pos = tf.Variable(tf.random.normal((sequence_length, running_units)), trainable=True)
        else:
            self.pos = tf.Variable(mp.FourierFeatures(tf.range(1000, dtype=tf.float32), 1, 150, running_units), trainable=False)
        self.alpha_pos = tf.Variable(0.1, trainable=True)
        
        # Beginning
        self.string_lookup = preprocessing.StringLookup(vocabulary=list(ALPHABET_UNMOD.keys()))
        #self.embedding = L.Embedding(len(ALPHABET_UNMOD), running_units, input_length=sequence_length)
        self.first = L.Dense(running_units)
        if prec_type in ['pretoken', 'inject_pre', 'inject_ffn']:
            self.charge_embedder = L.Dense(running_units) #mp.PrecursorToken(running_units, 64, 1, 15)
            self.ce_embedder = mp.PrecursorToken(running_units, running_units, 0.01, 1.5)
        
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
                norm_type=norm_type, 
                use_embed=True if prec_type in ['inject_pre', 'inject_ffn'] else False, 
                preembed=True if prec_type == 'inject_pre' else False, 
                is_cross=False
            )
            for _ in range(depth)
        ]

        # End
        penultimate_units = running_units if penultimate_units is None else penultimate_units
        self.penultimate = K.Sequential([
            L.Dense(penultimate_units),
            L.BatchNormalization(),
            L.ReLU()
        ])
        self.final = L.Dense(output_units, activation='sigmoid')

    def EmbedInputs(self, sequence, precursor_charge, collision_energy):
        length = sequence.shape[1]
        input_embedding = tf.one_hot(self.string_lookup(sequence), len(ALPHABET_UNMOD))
        if self.prec_type == 'embed_input':
            charge_emb = tf.tile(precursor_charge[:,None], [1, length, 1])
            ce_emb = tf.tile(collision_energy[:,None], [1, length, 1])
            input_embedding = tf.concat([input_embedding, charge_emb, ce_emb], axis=-1)
        
        return input_embedding

    def Main(self, x, tb_emb=None):
        out = x
        for i in range(len(self.main)):
            layer = self.main[i]

            if (self.inject_pos == "all") or \
               (self.inject_pos == "pre" and i == 0) or \
               (self.inject_pos == "post" and i == len(self.main) - 1):
                out = layer(out, temb=tb_emb)
            else:
                out = layer(out, None)

        return out

    def call(self, x, training=False):

        #out = self.EmbedInputs(x['sequence'], x['precursor_charge'], x['collision_energy'])
        out = self.EmbedInputs(x['_parsed_sequence'], x['precursor_charge_onehot'], x['collision_energy_aligned_normed'])

        out = self.first(out) + self.alpha_pos*self.pos[:out.shape[1]]
        tb_emb = None
        if self.prec_type == 'pretoken': 
            charge_ce_token = self.charge_embedder(x['precursor_charge']) + self.ce_embedder(x['collision_energy'])
            out = tf.concat([charge_ce_token[:,None], out], axis=1)
        elif self.prec_type in ['inject_pre', 'inject_ffn']:    # if chosen inject into transformer blocks
            charge_ce_embedding = tf.concat([
                self.charge_embedder(x['precursor_charge']),
                self.ce_embedder(x['collision_energy'])
            ], axis=-1)
            tb_emb = tf.nn.silu(charge_ce_embedding)
        out = self.Main(out, tb_emb=tb_emb)     # Transformer blocks
        out = self.penultimate(out)
        out = self.final(out)

        return tf.reduce_mean(out, axis=1)

