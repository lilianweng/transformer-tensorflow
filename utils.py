import os

import tensorflow as tf
from baselines.common.tf_util import display_var_info
from baselines.common.console_util import colorize

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))


def print_trainable_variables():
    display_var_info(tf.trainable_variables())


class BaseModelMixin:
    """Abstract object representing an Reader model.
    Code borrowed from: https://github.com/devsisters/DQN-tensorflow/blob/master/dqn/base.py
    with some modifications.
    """

    def __init__(self, model_name, tf_sess_config=None):
        self._saver = None
        self._writer = None
        self._model_name = model_name
        self._sess = None

        if tf_sess_config is None:
            tf_sess_config = {
                'allow_soft_placement': True,
                'intra_op_parallelism_threads': 8,
                'inter_op_parallelism_threads': 4,
            }
        self.tf_sess_config = tf_sess_config

    def get_variable_values(self):
        t_vars = tf.trainable_variables()
        vals = self.sess.run(t_vars)
        return {v.name: value for v, value in zip(t_vars, vals)}

    def save_checkpoint(self, step=None):
        print(colorize(" [*] Saving checkpoints...", "green"))
        ckpt_file = os.path.join(self.checkpoint_dir, self.model_name)
        self.saver.save(self.sess, ckpt_file, global_step=step)

    def load_checkpoint(self):
        print(colorize(" [*] Loading checkpoints...", "green"))
        ckpt_path = tf.train.latest_checkpoint(self.checkpoint_dir)
        print(self.checkpoint_dir)
        print("ckpt_path:", ckpt_path)

        if ckpt_path:
            # self._saver = tf.train.import_meta_graph(ckpt_path + '.meta')
            self.saver.restore(self.sess, ckpt_path)
            print(colorize(" [*] Load SUCCESS: %s" % ckpt_path, "green"))
            return True
        else:
            print(colorize(" [!] Load FAILED: %s" % self.checkpoint_dir, "red"))
            return False

    def _get_dir(self, dir_name):
        path = os.path.join(REPO_ROOT, dir_name, self.model_name)
        os.makedirs(path, exist_ok=True)
        return path

    @property
    def log_dir(self):
        return self._get_dir('logs')

    @property
    def checkpoint_dir(self):
        return self._get_dir('checkpoints')

    @property
    def model_dir(self):
        return self._get_dir('models')

    @property
    def tb_dir(self):
        # tensorboard
        return self._get_dir('tb')

    @property
    def model_name(self):
        assert self._model_name, "Not a valid model name."
        return self._model_name

    @property
    def saver(self):
        if self._saver is None:
            self._saver = tf.train.Saver(max_to_keep=5)
        return self._saver

    @property
    def writer(self):
        if self._writer is None:
            self._writer = tf.summary.FileWriter(self.tb_dir, self.sess.graph)
        return self._writer

    @property
    def sess(self):
        if self._sess is None:
            config = tf.ConfigProto(**self.tf_sess_config)
            self._sess = tf.Session(config=config)

        return self._sess
