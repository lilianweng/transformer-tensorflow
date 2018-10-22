import os

import tensorflow as tf
from baselines.common.console_util import colorize

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))


class BaseModelMixin:
    """Abstract object representing an Reader model.
    Code borrowed from: https://github.com/devsisters/DQN-tensorflow/blob/master/dqn/base.py
    with some modifications.
    """

    def __init__(self, model_name, saver_max_to_keep=5):
        self._saver = None
        self._saver_max_to_keep = saver_max_to_keep
        self._writer = None
        self._model_name = model_name
        self._sess = None

    def scope_vars(self, scope):
        res = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
        assert len(res) > 0
        print("Variables in scope '%s'" % scope)
        for v in res:
            print("\t" + str(v))
        return res

    def save_model(self, step=None):
        print(colorize(" [*] Saving checkpoints...", "green"))
        ckpt_file = os.path.join(self.checkpoint_dir, self.model_name)
        self.saver.save(self.sess, ckpt_file, global_step=step)

    def load_model(self):
        print(colorize(" [*] Loading checkpoints...", "green"))

        ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
        print(self.checkpoint_dir, ckpt)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            print(ckpt_name)
            fname = os.path.join(self.checkpoint_dir, ckpt_name)
            print(fname)
            self.saver.restore(self.sess, fname)
            print(colorize(" [*] Load SUCCESS: %s" % fname, "green"))
            return True
        else:
            print(colorize(" [!] Load FAILED: %s" % self.checkpoint_dir, "red"))
            return False

    @property
    def checkpoint_dir(self):
        ckpt_path = os.path.join(REPO_ROOT, 'checkpoints', self.model_name)
        os.makedirs(ckpt_path, exist_ok=True)
        return ckpt_path

    @property
    def model_name(self):
        assert self._model_name, "Not a valid model name."
        return self._model_name

    @property
    def saver(self):
        if self._saver is None:
            self._saver = tf.train.Saver(max_to_keep=self._saver_max_to_keep)
        return self._saver

    @property
    def writer(self):
        if self._writer is None:
            writer_path = os.path.join(REPO_ROOT, "logs", self.model_name)
            os.makedirs(writer_path, exist_ok=True)
            self._writer = tf.summary.FileWriter(writer_path, self.sess.graph)
        return self._writer

    @property
    def sess(self):
        if self._sess is None:
            tf_sess_configs = {
                'allow_soft_placement': True,
                'log_device_placement': False,
                'intra_op_parallelism_threads': 8,
                'inter_op_parallelism_threads': 4,
            }
            config = tf.ConfigProto(**tf_sess_configs)
            self._sess = tf.Session(config=config)

        return self._sess
