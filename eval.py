"""
Load a trained transformer model and evaluate it on the whole test set.
Author: Lilian Weng (lilian.wengweng@gmail.com)
        http://lilianweng.github.io/lil-log
        Oct 2018
"""
import click
from data import DatasetManager, PAD_ID, recover_sentence
from transformer import Transformer
from nltk.translate.bleu_score import corpus_bleu


@click.command()
@click.argument('model_name')
def eval(model_name):
    transformer = Transformer.load_model(model_name, is_training=False)
    transformer.print_trainable_variables()

    cfg = transformer.config

    dm = DatasetManager(cfg['dataset'])
    dm.maybe_download_data_files()

    refs = []
    hypos = []

    data_iter = dm.data_generator(
        cfg['train_params']['batch_size'],
        cfg['train_params']['seq_len'] + 1, data_type='test')
    for source_ids, target_ids in data_iter:
        pred_ids = transformer.predict(source_ids)
        refs += [[recover_sentence(sent_ids, dm.target_id2word)] for sent_ids in target_ids]
        hypos += [recover_sentence(sent_ids, dm.target_id2word) for sent_ids in pred_ids]
        print(f"Num. sentences processed: {len(hypos)}", end='\r', flush=True)
        if len(hypos) >= 100:
            break

    print()

    bleu_score = corpus_bleu(refs, hypos)
    results = dict(
        num_sentences=len(hypos),
        bleu_score=bleu_score * 100.,
    )
    print(results)


if __name__ == '__main__':
    eval()
