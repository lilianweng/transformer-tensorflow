"""
Check my blog post on attention and transformer:
    https://lilianweng.github.io/lil-log/2018/06/24/attention-attention.html

Implementations that helped me:
    https://github.com/Kyubyong/transformer/
    https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py
    http://nlp.seas.harvard.edu/2018/04/01/attention.html
"""
import numpy as np
import tensorflow as tf

from utils import BaseModelMixin
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from baselines.common.tf_util import display_var_info
from data import recover_sentence


class Transformer(BaseModelMixin):
    """
    See the architecture spec of Transformer in:
        Vaswani et al. Attention is All You Need. NIPS 2017.

    """

    def __init__(self, num_heads=8, d_model=512, d_ff=2048,
                 num_enc_layers=6, num_dec_layers=6,
                 drop_rate=0.1, use_label_smoothing=True, ls_epsilon=0.1,
                 model_name='transformer', tf_sess_config=None):
        assert d_model % num_heads == 0
        super().__init__(model_name, tf_sess_config=tf_sess_config)

        self.h = num_heads
        self.d_model = d_model
        self.d_ff = d_ff

        self.num_enc_layers = num_enc_layers
        self.num_dec_layers = num_dec_layers

        # Dropout regularization: added in every sublayer before layer_norm(...) and
        # applied to embedding + positional encoding.
        self.drop_rate = drop_rate

        # Label smoothing epsilon
        self.ls_epsilon = ls_epsilon
        self.use_label_smoothing = use_label_smoothing

        self._is_init = False
        self.step = 0  # training step.

        self.is_training = tf.placeholder(tf.bool, name='is_training')
        # The following variables will be initialized in build_model().
        self._input_id2word = None
        self._target_id2word = None
        self._pad_id = 0
        # The following variables will be constructed in build_model().
        self._input = None
        self._target = None
        self._output = None
        self._loss = None
        self._train_op = None

    def embedding(self, inp, vocab_size):
        embed_size = self.d_model
        embed_lookup = tf.get_variable("embed_lookup", [vocab_size, embed_size], tf.float32)
        out = tf.nn.embedding_lookup(embed_lookup, inp)
        return out

    def positional_encoding_embedding(self, inp):
        batch_size, seq_len = inp.shape.as_list()

        # Copy [0, 1, ..., `inp_size`] by `batch_size` times => matrix [batch, seq_len]
        pos_ind = tf.tile(tf.expand_dims(tf.range(seq_len), 0), [batch_size, 1])
        return self.embedding(pos_ind, seq_len)  # [batch, seq_len, d_model]

    def positional_encoding_sinusoid(self, inp):
        """
        PE(pos, 2i) = sin(pos / 10000^{2i/d_model})
        PE(pos, 2i+1) = cos(pos / 10000^{2i/d_model})
        """
        batch, seq_len = inp.shape.as_list()

        # Copy [0, 1, ..., `inp_size`] by `batch_size` times => matrix [batch, seq_len]
        pos_ind = tf.tile(tf.expand_dims(tf.range(seq_len), 0), [batch, 1])

        # Compute the arguments for sin and cos: pos / 10000^{2i/d_model})
        # `pos_enc` is a tensor of shape [seq_len, d_model]
        pos_enc = np.array([
            [pos / np.power(10000, 2. * i / self.d_model) for i in range(self.d_model)]
            for pos in range(seq_len)
        ])

        # Apply the cosine to even columns and sin to odds.
        pos_enc[:, 0::2] = np.sin(pos_enc[:, 0::2])  # dim 2i
        pos_enc[:, 1::2] = np.cos(pos_enc[:, 1::2])  # dim 2i+1

        # Convert to a tensor
        lookup_table = tf.convert_to_tensor(pos_enc, dtype=tf.float32)  # [seq_len, d_model]
        out = tf.nn.embedding_lookup(lookup_table, pos_ind)  # [batch, seq_len, d_model]
        return out

    def preprocess(self, inp, inp_vocab, scope):
        # Pre-processing: embedding + positional encoding
        with tf.variable_scope(scope):
            out = self.embedding(inp, inp_vocab)  # [batch, seq_len, d_model]
            out += self.positional_encoding_sinusoid(inp)
            out = tf.layers.dropout(out, rate=self.drop_rate, training=self.is_training)

        return out

    def layer_norm(self, inp):
        return tf.contrib.layers.layer_norm(inp)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """
        Assuming all the input tensors have three dimension:
        Args:
            Q: of shape (h * bs, q_size, d)
            K: of shape (h * bs, k_size, d)
            V: of shape (h * bs, k_size, d)
            mask: of shape (h * bs, q_size, k_size)
        """

        d = self.d_model // self.h
        assert d == Q.shape[-1] == K.shape[-1] == V.shape[-1]

        out = tf.matmul(Q, tf.transpose(K, [0, 2, 1]))  # [h*batch, q_size, k_size]

        if mask is not None:
            # masking out (0.0) => setting to -inf.
            out = tf.multiply(out, mask) + (1.0 - mask) * (-1e10)

        out = tf.nn.softmax(out / tf.sqrt(tf.cast(d, tf.float32)))  # [h * bs, q_size, k_size]
        out = tf.layers.dropout(out, training=self.is_training)
        out = tf.matmul(out, V)  # [h * bs, q_size, d]

        return out

    def multihead_attention(self, query, memory=None, mask=None, scope='attn'):
        """
        Args:
            query (tf.tensor): of shape (batch, q_size, q_embed_size)
            memory (tf.tensor): of shape (batch, m_size, k_embed_size)
            mask (tf.tensor): shape (batch, q_size, k_size)

        Returns:h
            a tensor of shape (bs, q_size, d_model)
        """
        if memory is None:
            memory = query

        with tf.variable_scope(scope):
            # Linear project to d_model dimension: [batch, q_size/k_size, d_model]
            Q = tf.layers.dense(query, self.d_model, activation=tf.nn.relu)
            K = tf.layers.dense(memory, self.d_model, activation=tf.nn.relu)
            V = tf.layers.dense(memory, self.d_model, activation=tf.nn.relu)

            # Split the matrix to multiple heads and then concatenate to have a larger
            # batch size: [h*batch, q_size/k_size, d_model/num_heads]
            Q_split = tf.concat(tf.split(Q, self.h, axis=2), axis=0)
            K_split = tf.concat(tf.split(K, self.h, axis=2), axis=0)
            V_split = tf.concat(tf.split(V, self.h, axis=2), axis=0)
            mask_split = tf.tile(mask, [self.h, 1, 1])

            # Apply scaled dot product attention
            out = self.scaled_dot_product_attention(Q_split, K_split, V_split, mask=mask_split)

            # Merge the multi-head back to the original shape
            out = tf.concat(tf.split(out, self.h, axis=0), axis=2)  # [bs, q_size, d_model]

            # The final linear layer and dropout.
            out = tf.layers.dense(out, self.d_model)
            out = tf.layers.dropout(out, rate=self.drop_rate, training=self.is_training)

        return out

    def feed_forwad(self, inp, scope='ff'):
        """
        Position-wise fully connected feed-forward network, applied to each position
        separately and identically. It can be implemented as (linear + ReLU + linear) or
        (conv1d + ReLU + conv1d).

        Args:
            inp (tf.tensor): shape [batch, length, d_model]
        """
        out = inp
        with tf.variable_scope(scope):
            out = tf.layers.dense(out, self.d_ff, activation=tf.nn.relu)
            out = tf.layers.dropout(out, rate=self.drop_rate, training=self.is_training)
            out = tf.layers.dense(out, self.d_model, activation=None)

        return out

    def construct_padding_mask(self, inp):
        """
        Args: Original input of word ids, shape [batch, seq_len]
        Returns: a mask of shape [batch, seq_len, seq_len], where <pad> is 0 and others are 1s.
        """
        seq_len = inp.shape.as_list()[1]
        mask = tf.cast(tf.not_equal(inp, self._pad_id), tf.float32)  # mask '<pad>'
        mask = tf.tile(tf.expand_dims(mask, 1), [1, seq_len, 1])
        return mask

    def construct_autoregressive_mask(self, target):
        """
        Args: Original target of word ids, shape [batch, seq_len]
        Returns: a mask of shape [batch, seq_len, seq_len].
        """
        batch_size, seq_len = target.shape.as_list()

        tri_matrix = np.zeros((seq_len, seq_len))
        tri_matrix[np.tril_indices(seq_len)] = 1

        mask = tf.convert_to_tensor(tri_matrix, dtype=tf.float32)
        masks = tf.tile(tf.expand_dims(mask, 0), [batch_size, 1, 1])  # copies
        return masks

    def encoder_layer(self, inp, input_mask, scope):
        """
        Args:
            inp: tf.tensor of shape (batch, seq_len, embed_size)
            input_mask: tf.tensor of shape (batch, seq_len, seq_len)
        """
        out = inp
        with tf.variable_scope(scope):
            # One multi-head attention + one feed-forword
            out = self.layer_norm(out + self.multihead_attention(out, mask=input_mask))
            out = self.layer_norm(out + self.feed_forwad(out))
        return out

    def encoder(self, inp, input_mask, scope='encoder'):
        """
        Args:
            inp (tf.tensor): shape (batch, seq_len, embed_size)
            input_mask (tf.tensor): shape (batch, seq_len, seq_len)
            scope (str):

        Returns:
        """
        out = inp  # now, (batch, seq_len, embed_size)
        with tf.variable_scope(scope):
            for i in range(self.num_enc_layers):
                out = self.encoder_layer(out, input_mask, f'enc_{i}')
        return out

    def decoder_layer(self, target, enc_out, input_mask, target_mask, scope):
        out = target
        with tf.variable_scope(scope):
            out = self.layer_norm(out + self.multihead_attention(
                out, mask=target_mask, scope='self_attn'))
            out = self.layer_norm(out + self.multihead_attention(
                out, memory=enc_out, mask=input_mask))
            out = self.layer_norm(out + self.feed_forwad(out))
        return out

    def decoder(self, target, enc_out, input_mask, target_mask, scope='decoder'):
        out = target
        with tf.variable_scope(scope):
            for i in range(self.num_enc_layers):
                out = self.decoder_layer(out, enc_out, input_mask, target_mask, f'dec_{i}')
        return out

    def label_smoothing(self, inp):
        """
        "... employed label smoothing of epsilon = 0.1 [36]. This hurts perplexity, as the model
        learns to be more unsure, but improves accuracy and BLEU score."

        Args:
            inp (tf.tensor): one-hot encoding vectors, [batch, seq_len, vocab_size]
        """
        vocab_size = inp.shape.as_list()[-1]
        smoothed = (1.0 - self.ls_epsilon) * inp + (self.ls_epsilon / vocab_size)
        return smoothed

    def build_model(self, input_id2word, target_id2word, pad_id, **train_params):
        """
        Args:
            input_id2word (dict)
            target_id2word (dict)
            pad_id (int): the id of '<pad>' symbol.

        Returns:
        """
        assert input_id2word[pad_id] == '<pad>'
        assert target_id2word[pad_id] == '<pad>'

        lr = train_params.get('lr', 0.0001)
        batch_size = train_params.get('batch_size', 32)
        seq_len = train_params.get('seq_len', 20)

        self._input_id2word = input_id2word
        self._target_id2word = target_id2word
        self._pad_id = np.int32(pad_id)

        input_vocab = len(input_id2word)
        target_vocab = len(target_id2word)

        with tf.variable_scope(self.model_name):
            self._input = tf.placeholder(tf.int32, shape=[batch_size, seq_len + 1], name='input')
            self._target = tf.placeholder(tf.int32, shape=[batch_size, seq_len + 1], name='target')

            # Add the offset on the input and target sentences.

            # For the input we remove the starting <s> to keep the seq len consistent.
            enc_inp = self._input[:, 1:]

            # For the decoder input, we remove the last element, as no more future prediction
            # is gonna be made based on it.
            dec_inp = self._target[:, :-1]
            dec_target = self._target[:, 1:]
            dec_target_ohe = tf.one_hot(dec_target, depth=target_vocab)
            if self.use_label_smoothing:
                dec_target_ohe = self.label_smoothing(dec_target_ohe)

            # The input mask only hides the <pad> symbol.
            input_mask = self.construct_padding_mask(enc_inp)

            # The target mask hides both <pad> and future words.
            target_mask = self.construct_padding_mask(dec_inp)
            target_mask *= self.construct_autoregressive_mask(dec_inp)

            # Input embedding + positional encoding
            inp_embed = self.preprocess(enc_inp, input_vocab, "input_preprocess")
            enc_out = self.encoder(inp_embed, input_mask)

            # Target embedding + positional encoding
            dec_inp_embed = self.preprocess(dec_inp, target_vocab, "target_preprocess")
            dec_out = self.decoder(dec_inp_embed, enc_out, input_mask, target_mask)

            # Make the prediction out of the decoder output.
            logits = tf.layers.dense(dec_out, target_vocab)  # [batch, target_vocab]
            probas = tf.nn.softmax(logits)
            self._output = tf.argmax(probas, axis=-1, output_type=tf.int32)
            print(logits.shape, probas.shape, self._output.shape)

            self._loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=dec_target_ohe))

            optim = tf.train.AdamOptimizer(learning_rate=lr)
            self._train_op = optim.minimize(self._loss)

        with tf.variable_scope(self.model_name + '_summary'):
            tf.summary.scalar('loss', self._loss)
            self.merged_summary = tf.summary.merge_all()

    def init(self):
        self.sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        self._is_init = True
        self.step = 0

    def done(self):
        self.writer.close()
        self.saver.save()  # Final checkpoint.

    def train(self, input_ids, target_ids):
        assert self._is_init, "Call .init() first."
        self.step += 1
        train_loss, summary, _ = self.sess.run(
            [self.loss, self.merged_summary, self.train_op],
            feed_dict={
                self.input_ph: input_ids.astype(np.int32),
                self.target_ph: target_ids.astype(np.int32),
                self.is_training: True,
            })
        self.writer.add_summary(summary, global_step=self.step)

        if self.step % 1000 == 0:
            # Save the model checkpoint every 1000 steps.
            self.save_model(step=self.step)

        return {'train_loss': train_loss, 'step': self.step}

    def predict(self, input_ids):
        assert list(input_ids.shape) == self.input_ph.shape.as_list()
        batch_size, seq_len = self.input_ph.shape.as_list()

        input_ids = input_ids.astype(np.int32)
        pred_ids = np.zeros(input_ids.shape, dtype=np.int32)

        # Predict one output a time autoregressively.
        for i in range(seq_len - 1):
            next_pred = self.sess.run(self._output, feed_dict={
                    self.input_ph: input_ids,
                    self.target_ph: pred_ids,
                    self.is_training: False,
                })
            # Only update the i-th column in one step.
            pred_ids[:, i] = next_pred[:, i]
            print(f"i={i}", pred_ids)

        return pred_ids

    def evaluate(self, input_ids, target_ids):
        pred_ids = self.predict(input_ids)

        refs = []
        hypos = []
        for truth, pred in zip(target_ids, pred_ids):
            truth_sent = recover_sentence(truth, self._target_id2word)
            pred_sent = recover_sentence(pred, self._target_id2word)

            refs.append([truth_sent])
            hypos.append(pred_sent)

        # Print the last pair for fun.
        source_sent = recover_sentence(input_ids[-1], self._input_id2word)
        print("[Source]", source_sent)
        print("[Truth]", truth_sent)
        print("[Translated]", pred_sent)

        smoothie = SmoothingFunction().method4
        bleu_score = corpus_bleu(refs, hypos, smoothing_function=smoothie)
        return {'bleu_score': bleu_score}

    # ============================= Utils ===============================

    def print_trainable_variables(self):
        t_vars = tf.trainable_variables(scope=self.model_name)
        display_var_info(t_vars)

    def _check_variable(self, v, name):
        if v is None:
            raise ValueError(f"Call build_model() to initialize {name}.")
        return v

    @property
    def input_ph(self):
        return self._check_variable(self._input, 'input placeholder')

    @property
    def target_ph(self):
        return self._check_variable(self._target, 'target placeholder')

    @property
    def train_op(self):
        return self._check_variable(self._train_op, 'train_op')

    @property
    def loss(self):
        return self._check_variable(self._loss, 'loss')
