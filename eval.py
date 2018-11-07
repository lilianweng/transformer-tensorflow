"""
Load a trained transformer model and evaluate it on the test data.

Author: Lilian Weng (lilian.wengweng@gmail.com)
        http://lilianweng.github.io/lil-log
        Oct 2018
"""
import click
import numpy as np
from data import DatasetManager, recover_sentence, PAD_ID
from transformer import Transformer
from nltk.translate.bleu_score import corpus_bleu


@click.command()
@click.argument('model_name')
@click.option('--file-prefix', '-f', type=str, default=None)
def eval(model_name, file_prefix):
    transformer = Transformer.load_model(model_name, is_training=False)

    cfg = transformer.config
    batch_size = cfg['train_params']['batch_size']
    seq_len = cfg['train_params']['seq_len'] + 1
    print(f'batch_size:{batch_size} seq_len:{seq_len}')

    dm = DatasetManager(cfg['dataset'])
    dm.maybe_download_data_files()
    data_iter = dm.data_generator(batch_size, seq_len, data_type='test',
                                  file_prefix=file_prefix, epoch=1)

    refs = []
    hypos = []
    for source_ids, target_ids in data_iter:
        valid_size = len(source_ids)

        if valid_size < batch_size:
            source_ids = np.array(list(source_ids) + [[PAD_ID] * seq_len] * (batch_size - source_ids.shape[0]))
            target_ids = np.array(list(target_ids) + [[PAD_ID] * seq_len] * (batch_size - target_ids.shape[0]))

        pred_ids = transformer.predict(source_ids)

        refs += [[recover_sentence(sent_ids, dm.target_id2word)]
                 for sent_ids in target_ids[:valid_size]]
        hypos += [recover_sentence(sent_ids, dm.target_id2word)
                  for sent_ids in pred_ids[:valid_size]]
        print(f"Num. sentences processed: {len(hypos)}", end='\r', flush=True)

    print()

    bleu_score = corpus_bleu(refs, hypos)
    results = dict(
        num_sentences=len(hypos),
        bleu_score=bleu_score * 100.,
    )

    # Sample a few translated sentences.
    indices = np.random.choice(list(range(len(hypos))), size=10, replace=False)
    for i in indices:
        print(f"Source: '{refs[i][0]}' ==> Target: '{hypos[i]}'.")

    print(results)


if __name__ == '__main__':
    eval()
