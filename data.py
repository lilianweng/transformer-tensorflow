import os
import sys
import urllib.parse
import numpy as np
import random

# IDs of special characters.
PAD_ID = 0
UNKNOWN_ID = 1
START_ID = 2
END_ID = 3


class DatasetManager:
    dataset_config_dict = {
        'iwslt15': {
            'source_lang': 'en',
            'target_lang': 'vi',
            'url': "https://nlp.stanford.edu/projects/nmt/data/iwslt15.en-vi/",
            'files': ['train.en', 'train.vi', 'tst2012.en', 'tst2012.en',
                      'tst2013.en', 'tst2013.vi', 'vocab.en', 'vocab.vi']
        }

    }

    def __init__(self, name, base_data_dir='/tmp/'):
        assert name in self.dataset_config_dict

        self.name = name
        self.data_config = self.dataset_config_dict[name]
        self.data_dir = os.path.join(base_data_dir, name)
        os.makedirs(self.data_dir, exist_ok=True)

    def _download_data_from_url(self, download_url):
        filename = download_url.split('/')[-1]
        filepath = os.path.join(self.data_dir, filename)

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

    def maybe_download_data_files(self):
        """Download and extract the file from Stanford NLP website.
        """
        for filename in self.data_config['files']:
            self._download_data_from_url(
                urllib.parse.urljoin(self.data_config['url'], filename)
            )

        return os.listdir(self.data_dir)


def load_vocab(vocab_file):
    # The first three words in both vocab files are special characters:
    # <unk>: unknown word.
    # <s>: start of a sentence.
    # </s>: # end of a sentence.
    # In addition, we add <pad> as a place holder for a padding space.
    words = list(map(lambda w: w.strip().lower(), open(vocab_file)))
    words.insert(0, '<pad>')
    words = words[:4] + list(set(words[4:]))  # Keep the special characters on top.
    word2id = {word: i for i, word in enumerate(words)}
    id2word = {i: word for i, word in enumerate(words)}

    assert id2word[PAD_ID] == '<pad>'
    assert id2word[UNKNOWN_ID] == '<unk>'
    assert id2word[START_ID] == '<s>'
    assert id2word[END_ID] == '</s>'

    return word2id, id2word


def sentence_pair_iterator(file1, file2, word2id1, word2id2, seq_len):
    """
    The sentence is discarded if it is longer than `seq_len`; otherwise we pad it with
    '<pad>' to make it to have the exact length `seq_len`.

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
        # If the sentence is not long enough, extend with '<pad>' symbols.
        word_ids = [START_ID] + word_ids + [END_ID]
        word_ids += [PAD_ID] * max(0, seq_len - len(word_ids))
        return word_ids

    line_pairs = list(zip(open(file1), open(file2)))
    random.shuffle(line_pairs)

    for l1, l2 in line_pairs:
        sent1 = parse_line(l1, word2id1)
        sent2 = parse_line(l2, word2id2)
        if len(sent1) == len(sent2) == seq_len:
            yield sent1, sent2


def data_generator(batch_size, seq_len, data_dir="/tmp/iwslt15/", prefix='train'):
    # Load vocabulary
    en2id, id2en = load_vocab(os.path.join(data_dir, 'vocab.en'))
    vi2id, id2vi = load_vocab(os.path.join(data_dir, 'vocab.vi'))

    batch_en, batch_vi = [], []
    while True:
        for ids_en, ids_vi in sentence_pair_iterator(
                os.path.join(data_dir, prefix + '.en'),
                os.path.join(data_dir, prefix + '.vi'),
                en2id, vi2id, seq_len
        ):
            batch_en.append(ids_en)
            batch_vi.append(ids_vi)

            if len(batch_en) == batch_size:
                yield np.array(batch_en).copy(), np.array(batch_vi).copy()
                batch_en, batch_vi = [], []


def recover_sentence(sent_ids, id2word):
    words = list(map(lambda i: id2word.get(i, '<unk>'), sent_ids))

    # Then remove tailing <pad>
    i = len(words) - 1
    while i >= 0 and words[i] == '<pad>':
        i -= 1
    words = words[:i + 1]
    return ' '.join(words)
