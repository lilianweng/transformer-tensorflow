import os

import tensorflow as tf
from baselines.common.console_util import colorize

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))


class BaseModelMixin:
    """Abstract object representing an Reader model.
    Code borrowed from: https://github.com/devsisters/DQN-tensorflow/blob/master/dqn/base.py
    with some modifications.
    """

    def __init__(self, model_name, saver_max_to_keep=5, tf_sess_config=None):
        print("Model name:", model_name)

        self._saver = None
        self._saver_max_to_keep = saver_max_to_keep
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

    def scope_vars(self, scope):
        res = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
        assert len(res) > 0
        print("Variables in scope '%s'" % scope)
        for v in res:
            print("\t" + str(v))
        return res

    def save_checkpoint(self, step=None):
        print(colorize(" [*] Saving checkpoints...", "green"))
        ckpt_file = os.path.join(self.checkpoint_dir, self.model_name)
        self.saver.save(self.sess, ckpt_file, global_step=step)

    def load_checkpoint(self):
        print(colorize(" [*] Loading checkpoints...", "green"))
        ckpt_path = tf.train.latest_checkpoint(self.checkpoint_dir)
        print(self.checkpoint_dir, ckpt_path)

        if ckpt_path:
            self._saver = tf.train.import_meta_graph(ckpt_path + '.meta')
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
            self._saver = tf.train.Saver(max_to_keep=self._saver_max_to_keep)
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
