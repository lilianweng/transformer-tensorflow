"""
Train a transformer model.
Author: Lilian Weng (lilian.wengweng@gmail.com)
        http://lilianweng.github.io/lil-log
        Oct 2018
"""

import click
import time

from baselines import logger
from data import *
from transformer import *
from utils import print_trainable_variables


@click.command()
@click.option('--seq-len', type=int, default=10, show_default=True, help="Input sequence length.")
@click.option('--d-model', type=int, default=512, show_default=True, help="d_model")
@click.option('--d-ff', type=int, default=2048, show_default=True, help="d_ff")
@click.option('--n-head', type=int, default=8, show_default=True, help="n_head")
@click.option('--batch-size', type=int, default=256, show_default=True, help="Batch size")
@click.option('--max-steps', type=int, default=300_000, show_default=True, help="Max train steps.")
@click.option('--dataset', type=click.Choice(['iwslt15', 'wmt14', 'wmt15']),
              default='iwslt15', show_default=True, help="Which translation dataset to use.")
def train(seq_len, d_model, d_ff, n_head, batch_size, max_steps, dataset):
    dm = DatasetManager(dataset)
    dm.maybe_download_data_files()
    dm.load_vocab()

    train_params = dict(
        learning_rate=1e-4,
        batch_size=batch_size,
        seq_len=seq_len,
        max_steps=max_steps,
    )

    tf_sess_config = dict(
        allow_soft_placement=True,
        intra_op_parallelism_threads=8,
        inter_op_parallelism_threads=4,
    )

    model_name = f'transformer-{dataset}-seq{seq_len}-d{d_model}-head{n_head}-{int(time.time())}'
    transformer = Transformer(
        num_heads=n_head,
        d_model=d_model,
        d_ff=d_ff,
        model_name=model_name,
        tf_sess_config=tf_sess_config
    )
    transformer.build_model(dataset, dm.source_id2word, dm.target_id2word, PAD_ID, **train_params)
    print_trainable_variables()

    train_data_iter = dm.data_generator(batch_size, seq_len + 1, data_type='train')
    test_data_iter = dm.data_generator(batch_size, seq_len + 1, data_type='test')
    logger.configure(dir=transformer.log_dir, format_strs=['stdout', 'csv'])

    transformer.init()  # step = 0
    while transformer.step < max_steps:
        input_ids, target_ids = next(train_data_iter)
        meta = transformer.train(input_ids, target_ids)
        for k, v in meta.items():
            logger.logkv(k, v)

        if transformer.step % 100 == 0:
            test_inp_ids, test_target_ids = next(test_data_iter)
            meta = transformer.evaluate(test_inp_ids, test_target_ids)
            for k, v in meta.items():
                logger.logkv('test_' + k, v)
            logger.dumpkvs()

    transformer.done()


if __name__ == '__main__':
    train()
