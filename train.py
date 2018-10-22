import click
import json
from baselines import logger
from data import *
from transformer import *


@click.command()
@click.option('--seq-len', type=int, default=20, help="Input sequence length.")
@click.option('--batch-size', type=int, default=64, help="Max training steps.")
@click.option('--max-steps', type=int, default=100000, help="Max training steps.")
def train(seq_len=20, batch_size=64, max_steps=100000):
    data_dir = '/tmp/iwslt15/'
    maybe_download_data_files(data_dir)

    # Load vocabulary first.
    en2id, id2en = load_vocab(os.path.join(data_dir, 'vocab.en'))
    vi2id, id2vi = load_vocab(os.path.join(data_dir, 'vocab.vi'))
    print("English vocabulary size:", len(en2id))
    print("Vietnamese vocabulary size:", len(vi2id))

    train_params = dict(
        learning_rate=1e-4,
        batch_size=batch_size,
        seq_len=seq_len,
        max_steps=max_steps,
    )

    transformer = Transformer(num_heads=4, d_model=128)
    transformer.build_model(id2en, vi2id, **train_params)
    transformer.print_trainable_variables()

    step = 0
    test_data_iter = data_generator(batch_size, seq_len, data_dir=data_dir, file_prefix='tst2013')

    transformer.init()
    while step < max_steps:
        for input_ids, target_ids in data_generator(batch_size, seq_len, data_dir=data_dir):
            meta = transformer.train(input_ids, target_ids)
            step += 1

            logger.logkv('step', step)
            logger.logkv('train_loss', meta['loss'])

            if step % 10 == 0:
                test_inp_ids, test_target_ids = next(test_data_iter)
                meta = transformer.evaluate(test_inp_ids, test_target_ids)
                logger.logkv('test_loss', meta['loss'])
                logger.logkv('test_bleu', meta['bleu'])
                logger.dumpkvs()


if __name__ == '__main__':
    train()
