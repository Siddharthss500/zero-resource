import argparse
import random
import torch
import os
import itertools

import Model
from loader import load_sentences, update_tag_scheme, word_mapping, augment_with_pretrained

def main(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Setup torch device
    device = torch.device('cuda' if args.device else 'cpu')

    # Check validity of arguments
    assert os.path.isfile(args.train)
    assert os.path.isfile(args.valid)
    assert os.path.isfile(args.test)
    assert args.tag_scheme in ['iob', 'iobes']
    assert args.char_dim > 0 or args.word_dim > 0

    # Initialize model
    # model = Model(parameters=args)

    # Data parameters
    lower = args.lower
    zeros = args.zeros
    tag_scheme = args.tag_scheme

    # Load sentences
    train_sentences = load_sentences(args.train, lower, zeros)
    valid_sentences = load_sentences(args.valid, lower, zeros)
    test_sentences = load_sentences(args.test, lower, zeros)

    # Use selected tagging scheme
    update_tag_scheme(train_sentences, tag_scheme)
    update_tag_scheme(valid_sentences, tag_scheme)
    update_tag_scheme(test_sentences, tag_scheme)

    # Create dictionary/mapping of words
    dico_words_train = word_mapping(train_sentences, lower)[0]
    words = list(itertools.chain.from_iterable([[w[0] for w in s] for s in valid_sentences + test_sentences]))
    dico_words, word_to_id, id_to_word = augment_with_pretrained(dico_words_train.copy(), words)

    # Index data

    # Save mappings

    # Build model

    # Train network

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--train', type=str, default='data/eng.train',
                        help='train file path')
    parser.add_argument('--valid', type=str, default='data/eng.testa',
                        help='validation file path')
    parser.add_argument('--test', type=str, default='data/eng.testb',
                        help='test file path')
    parser.add_argument('--tag_scheme', type=str, default="iobes",
                        help='Tagging scheme (IOB or IOBES)')
    parser.add_argument('--lower', type=int, default=0,
                        help='Lowercase words')
    parser.add_argument('--zeros', type=int, default=0,
                        help='Replace digits with zeros')
    parser.add_argument('--char_dim', type=int, default=25,
                        help='Dimension of character embeddings')
    parser.add_argument('--char_lstm_dim', type=int, default=25,
                        help='Dimension of LSTM character embeddings')
    parser.add_argument('--word_dim', type=int, default=100,
                        help='Dimension of word embeddings')
    parser.add_argument('--word_lstm_dim', type=int, default=100,
                        help='Dimension of LSTM word embeddings')
    parser.add_argument('--cap_dim', type=int, default=0,
                        help='Capitalization feature dimension')
    parser.add_argument('--seed', type=int, default=42,
                        help='seed for model')
    parser.add_argument('--device', type=int, default=1,
                        help='Run model on cpu(0) or gpu(1 default)')

    args = parser.parse_args()
    main(args)