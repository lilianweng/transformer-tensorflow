import os
import sys
import urllib
import numpy as np

# IDs of special characters.
EMPTY_ID = 0
UNKNOWN_ID = 1


def download_data_from_url(download_url, data_dir):
    filename = download_url.split('/')[-1]
    filepath = os.path.join(data_dir, filename)

    if not os.path.exists(filepath):
        # If the file does not exist, download it.
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (
                filename, float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        filepath, _ = urllib.request.urlretrieve(download_url, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')

    return filepath


def maybe_download_data_files(data_dir):
    """Download and extract the file from Stanford NLP website.
    """
    os.makedirs(data_dir, exist_ok=True)
    site_prefix = "https://nlp.stanford.edu/projects/nmt/data/iwslt15.en-vi/"
    filenames = ['train.en', 'train.vi', 'tst2012.en', 'tst2012.en',
                 'tst2013.en', 'tst2013.vi', 'vocab.en', 'vocab.vi']
    for filename in filenames:
        download_data_from_url(urllib.parse.urljoin(site_prefix, filename), data_dir)

    return os.listdir(data_dir)


def load_vocab(vocab_file):
    # The first three words in both vocab files are special characters:
    # <unk>: unknown word.
    # <s>: start of a sentence.
    # </s>: # end of a sentence.
    # In addition, we add <empty> as a place holder for empty space.
    words = list(map(lambda w: w.strip().lower(), open(vocab_file)))
    words.insert(0, '<empty>')
    words = words[:4] + list(set(words[4:]))  # Keep the special characters on top.
    word2id = {word: i for i, word in enumerate(words)}
    id2word = {i: word for i, word in enumerate(words)}

    assert id2word[EMPTY_ID] == '<empty>'
    assert id2word[UNKNOWN_ID] == '<unk>'
    return word2id, id2word


def sentence_pair_iterator(file1, file2, word2id1, word2id2, seq_len):
    """
    The sentence is discarded if it is longer than `seq_len`; otherwise we pad it with
    '<empty>' to make it to have the exact length `seq_len`.

    Args:
        file1 (str): training data in language 1.
        file2 (str): training data in language 2. Lines should match lines in `file1`.
        word2id1 (dict): word-ID mapping for language 1.
        word2id2 (dict):: word-ID mapping for language 2.
        seq_len (int): max sequence length.

    Returns: a tuple of (a list of word id for language 1,
                         a list of word id for language 2)
    """

    def parse_line(line, word2id):
        line = line.strip().lower().split()
        word_ids = [word2id.get(w, UNKNOWN_ID) for w in line]
        # If the sentence is not long enough, pad empty symbols.
        word_ids += [EMPTY_ID] * max(0, seq_len - len(word_ids))
        return word_ids

    for l1, l2 in zip(open(file1), open(file2)):
        sent1 = parse_line(l1, word2id1)
        sent2 = parse_line(l2, word2id2)
        if len(sent1) == len(sent2) == seq_len:
            yield sent1, sent2


def data_generator(batch_size, seq_len, data_dir="/tmp/iwslt15/", file_prefix='train'):
    # Load vocabulary
    en2id, id2en = load_vocab(os.path.join(data_dir, 'vocab.en'))
    vi2id, id2vi = load_vocab(os.path.join(data_dir, 'vocab.vi'))

    batch_en, batch_vi = [], []
    while True:
        for ids_en, ids_vi in sentence_pair_iterator(
                os.path.join(data_dir, file_prefix + '.en'),
                os.path.join(data_dir, file_prefix + '.vi'),
                en2id, vi2id, seq_len
        ):
            batch_en.append(ids_en)
            batch_vi.append(ids_vi)

            if len(batch_en) == batch_size:
                yield np.array(batch_en).copy(), np.array(batch_vi).copy()
                batch_en, batch_vi = [], []
