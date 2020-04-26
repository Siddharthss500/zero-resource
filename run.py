"""
Usage:
    run.py train TRAIN SENT_VOCAB TAG_VOCAB_NER TAG_VOCAB_ENTITY [options]
    run.py test TEST SENT_VOCAB TAG_VOCAB_NER TAG_VOCAB_ENTITY MODEL [options]

Options:
    --dropout-rate=<float>              dropout rate [default: 0.5]
    --embed-size=<int>                  size of word embedding [default: 256]
    --hidden-size=<int>                 size of hidden state [default: 256]
    --batch-size=<int>                  batch-size [default: 32]
    --max-epoch=<int>                   max epoch [default: 10]
    --clip_max_norm=<float>             clip max norm [default: 5.0]
    --lr=<float>                        learning rate [default: 0.001]
    --log-every=<int>                   log every [default: 10]
    --validation-every=<int>            validation every [default: 250]
    --patience-threshold=<float>        patience threshold [default: 0.98]
    --max-patience=<int>                time of continuous worse performance to decay lr [default: 4]
    --max-decay=<int>                   time of lr decay to early stop [default: 4]
    --lr-decay=<float>                  decay rate of lr [default: 0.5]
    --model-save-path=<file>            model save path [default: ./model/model.pth]
    --optimizer-save-path=<file>        optimizer save path [default: ./model/optimizer.pth]
    --cuda                              use GPU
"""

from docopt import docopt
from vocab import Vocab
import time
import torch
import torch.nn as nn
import bilstm_crf
import utils
import random

import codecs
from collections import Counter
import json
import fasttext
import numpy as np

def train(args, weights_matrix):
    """ Training BiLSTMCRF model
    Args:
        args: dict that contains options in command
    """
    sent_vocab = Vocab.load(args['SENT_VOCAB'])
    tag_vocab_ner = Vocab.load(args['TAG_VOCAB_NER'])
    tag_vocab_entity = Vocab.load(args['TAG_VOCAB_ENTITY'])
    train_data, dev_data = utils.generate_train_dev_dataset(args['TRAIN'], sent_vocab, tag_vocab_ner, tag_vocab_entity)
    print('num of training examples: %d' % (len(train_data)))
    print('num of development examples: %d' % (len(dev_data)))

    max_epoch = int(args['--max-epoch'])
    log_every = int(args['--log-every'])
    validation_every = int(args['--validation-every'])
    model_save_path = args['--model-save-path']
    optimizer_save_path = args['--optimizer-save-path']
    min_dev_loss = float('inf')
    device = torch.device('cuda' if args['--cuda'] else 'cpu')
    patience, decay_num = 0, 0

    model = bilstm_crf.BiLSTMCRF(weights_matrix, sent_vocab, tag_vocab_ner, tag_vocab_entity, float(args['--dropout-rate']), int(args['--embed-size']),
                                 int(args['--hidden-size'])).to(device)
    print(model)
    # for name, param in model.named_parameters():
    #     if 'weight' in name:
    #         nn.init.normal_(param.data, 0, 0.01)
    #     else:
    #         nn.init.constant_(param.data, 0)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(args['--lr']))
    train_iter = 0  # train iter num
    record_loss_sum, record_tgt_word_sum, record_batch_size = 0, 0, 0  # sum in one training log
    cum_loss_sum, cum_tgt_word_sum, cum_batch_size = 0, 0, 0  # sum in one validation log
    record_start, cum_start = time.time(), time.time()

    print('start training...')
    for epoch in range(max_epoch):
        for sentences, tags_ner, tags_entity in utils.batch_iter(train_data, batch_size=int(args['--batch-size'])):
            train_iter += 1
            current_batch_size = len(sentences)
            sentences, sent_lengths = utils.pad(sentences, sent_vocab[sent_vocab.PAD], device)
            tags_ner, _ = utils.pad(tags_ner, tag_vocab_ner[tag_vocab_ner.PAD], device)
            tags_entity, _ = utils.pad(tags_entity, tag_vocab_entity[tag_vocab_entity.PAD], device)

            # back propagation
            optimizer.zero_grad()
            batch_loss = model(sentences, tags_ner, tags_entity, sent_lengths)  # shape: (b,)
            loss = batch_loss.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(args['--clip_max_norm']))
            optimizer.step()

            record_loss_sum += batch_loss.sum().item()
            record_batch_size += current_batch_size
            record_tgt_word_sum += sum(sent_lengths)

            cum_loss_sum += batch_loss.sum().item()
            cum_batch_size += current_batch_size
            cum_tgt_word_sum += sum(sent_lengths)

            if train_iter % log_every == 0:
                print('log: epoch %d, iter %d, %.1f words/sec, avg_loss %f, time %.1f sec' %
                      (epoch + 1, train_iter, record_tgt_word_sum / (time.time() - record_start),
                       record_loss_sum / record_batch_size, time.time() - record_start))
                record_loss_sum, record_batch_size, record_tgt_word_sum = 0, 0, 0
                record_start = time.time()

            if train_iter % validation_every == 0:
                print('dev: epoch %d, iter %d, %.1f words/sec, avg_loss %f, time %.1f sec' %
                      (epoch + 1, train_iter, cum_tgt_word_sum / (time.time() - cum_start),
                       cum_loss_sum / cum_batch_size, time.time() - cum_start))
                cum_loss_sum, cum_batch_size, cum_tgt_word_sum = 0, 0, 0

                dev_loss = cal_dev_loss(model, dev_data, 64, sent_vocab, tag_vocab_ner, tag_vocab_entity, device)
                if dev_loss < min_dev_loss * float(args['--patience-threshold']):
                    min_dev_loss = dev_loss
                    model.save(model_save_path)
                    torch.save(optimizer.state_dict(), optimizer_save_path)
                    print('Reached %d epochs, Save result model to %s' % (epoch, model_save_path))
                    patience = 0
                    # Save the word embeddings
                    print("Saving the model")
                    params = torch.load(model_save_path, map_location=lambda storage, loc: storage)
                    new_weights_matrix = params['state_dict']['embedding.weight']
                    b = new_weights_matrix.tolist()
                    file_path = "./data/weights_matrix.json"
                    json.dump(b, codecs.open(file_path, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)
                else:
                    patience += 1
                    if patience == int(args['--max-patience']):
                        decay_num += 1
                        if decay_num == int(args['--max-decay']):
                            return
                        lr = optimizer.param_groups[0]['lr'] * float(args['--lr-decay'])
                        model = bilstm_crf.BiLSTMCRF.load(weights_matrix, model_save_path, device)
                        optimizer.load_state_dict(torch.load(optimizer_save_path))
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr
                        patience = 0
                print('dev: epoch %d, iter %d, dev_loss %f, patience %d, decay_num %d' %
                      (epoch + 1, train_iter, dev_loss, patience, decay_num))
                cum_start = time.time()
                if train_iter % log_every == 0:
                    record_start = time.time()
    # model.save(model_save_path)
    # print('Reached %d epochs, Save result model to %s' % (max_epoch, model_save_path))


def test(args, weights_matrix):
    """ Testing the model
    Args:
        args: dict that contains options in command
    """
    sent_vocab = Vocab.load(args['SENT_VOCAB'])
    tag_vocab = Vocab.load(args['TAG_VOCAB_NER'])
    sentences, tags = utils.read_corpus(args['TEST'])
    sentences = utils.words2indices(sentences, sent_vocab)

    # # Convert to binary tags (if there is a tag or not)
    tags_entity = utils.entity_or_not(tags)

    # Convert from IOBES to IOB
    tags = iobes_iob(tags)

    tags = utils.words2indices(tags, tag_vocab)
    test_data = list(zip(sentences, tags, tags_entity))
    print('num of test samples: %d' % (len(test_data)))

    device = torch.device('cuda' if args['--cuda'] else 'cpu')
    model = bilstm_crf.BiLSTMCRF.load(weights_matrix, args['MODEL'], device)
    print('start testing...')
    print('using device', device)

    start = time.time()
    n_iter, num_words = 0, 0
    tp, fp, fn = 0, 0, 0

    model.eval()
    with torch.no_grad():
        for sentences, tags, tags_entity in utils.batch_iter(test_data, batch_size=int(args['--batch-size']), shuffle=False):
            sentences, sent_lengths = utils.pad(sentences, sent_vocab[sent_vocab.PAD], device)
            predicted_tags = model.predict(sentences, sent_lengths)
            n_iter += 1
            num_words += sum(sent_lengths)
            for tag, predicted_tag in zip(tags, predicted_tags):
                current_tp, current_fp, current_fn = cal_statistics(tag, predicted_tag, tag_vocab)
                tp += current_tp
                fp += current_fp
                fn += current_fn
            if n_iter % int(args['--log-every']) == 0:
                print('log: iter %d, %.1f words/sec, precision %f, recall %f, f1_score %f, time %.1f sec' %
                      (n_iter, num_words / (time.time() - start), tp / (tp + fp), tp / (tp + fn),
                       (2 * tp) / (2 * tp + fp + fn), time.time() - start))
                num_words = 0
                start = time.time()
    print('tp = %d, fp = %d, fn = %d' % (tp, fp, fn))
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = (2 * tp) / (2 * tp + fp + fn)
    print('Precision: %f, Recall: %f, F1 score: %f' % (precision, recall, f1_score))


def cal_dev_loss(model, dev_data, batch_size, sent_vocab, tag_vocab_ner, tag_vocab_entity, device):
    """ Calculate loss on the development data
    Args:
        model: the model being trained
        dev_data: development data
        batch_size: batch size
        sent_vocab: sentence vocab
        tag_vocab: tag vocab
        device: torch.device on which the model is trained
    Returns:
        the average loss on the dev data
    """
    is_training = model.training
    model.eval()
    loss, n_sentences = 0, 0
    with torch.no_grad():
        for sentences, tags_ner, tags_entity in utils.batch_iter(dev_data, batch_size, shuffle=False):
            sentences, sent_lengths = utils.pad(sentences, sent_vocab[sent_vocab.PAD], device)
            tags_ner, _ = utils.pad(tags_ner, tag_vocab_ner[sent_vocab.PAD], device)
            tags_entity, _ = utils.pad(tags_entity, tag_vocab_entity[sent_vocab.PAD], device)
            batch_loss = model(sentences, tags_ner, tags_entity, sent_lengths)  # shape: (b,)
            loss += batch_loss.sum().item()
            n_sentences += len(sentences)
    model.train(is_training)
    return loss / n_sentences


def cal_statistics(tag, predicted_tag, tag_vocab):
    """ Calculate TN, FN, FP for the given true tag and predicted tag.
    Args:
        tag (list[int]): true tag
        predicted_tag (list[int]): predicted tag
        tag_vocab: tag vocab
    Returns:
        tp: true positive
        fp: false positive
        fn: false negative
    """
    tp, fp, fn = 0, 0, 0

    def func(tag1, tag2):
        a, b, i = 0, 0, 0
        while i < len(tag1):
            if tag1[i] == tag_vocab['O']:
                i += 1
                continue
            begin, end = i, i
            while end + 1 < len(tag1) and tag1[end + 1] != tag_vocab['O']:
                end += 1
            equal = True
            for j in range(max(0, begin - 1), min(len(tag1), end + 2)):
                if tag1[j] != tag2[j]:
                    equal = False
                    break
            a, b = a + equal, b + 1 - equal
            i = end + 1
        return a, b
    t, f = func(tag, predicted_tag)
    tp += t
    fn += f
    t, f = func(predicted_tag, tag)
    fp += f
    return tp, fp, fn

def preprocess_data(args, parameter='TRAIN'):
    """
    Load sentences. A line must contain at least a word and its tag.
    Sentences are separated by empty lines.
    """
    sentences = []
    sentence = []
    for line in codecs.open(args[parameter], 'r', 'utf8'):
        line = line.rstrip()
        if not line:
            if len(sentence) > 0:
                if 'DOCSTART' not in sentence[0][0]:
                    sentences.append(sentence)
                sentence = []
        else:
            word = line.split()
            assert len(word) >= 2
            sentence.append(word)
    if len(sentence) > 0:
        if 'DOCSTART' not in sentence[0][0]:
            sentences.append(sentence)

    tags_ner = ['<START>',  '<END>', '<PAD>', '-DOCSTART-']
    tags_entity = ['<START>',  '<END>', '<PAD>', '-DOCSTART-']
    words = ['<START>',  '<END>', '<PAD>', '-DOCSTART-']

    for sentence in sentences:
        for sent in sentence:
            words.append(sent[0])
            tags_ner.append(sent[1])
            if sent[1] == 'O':
                tags_entity.append('O')
            else:
                tags_entity.append('Y')
    unique_tags_ner = list(Counter(tags_ner).keys())
    unique_tags_entity = list(Counter(tags_entity).keys())
    unique_words = list(Counter(words).keys())

    return unique_tags_ner, unique_tags_entity, unique_words

def create_vocab(unique_tags_ner, unique_tags_entity, unique_words):
    # For tags NER
    unique_tags_dict = {unique_tags_ner[i]: i for i in range(len(unique_tags_ner))}
    tag_vocab = {"word2id": unique_tags_dict, "id2word": unique_tags_ner}
    json_object = json.dumps(tag_vocab)
    with open("./vocab/tag_vocab_ner.json", "w") as outfile:
        outfile.write(json_object)
    # For tags entity
    unique_tags_dict = {unique_tags_entity[i]: i for i in range(len(unique_tags_entity))}
    tag_vocab = {"word2id": unique_tags_dict, "id2word": unique_tags_entity}
    json_object = json.dumps(tag_vocab)
    with open("./vocab/tag_vocab_entity.json", "w") as outfile:
        outfile.write(json_object)
    # For words
    unique_words_dict = {unique_words[i]: i for i in range(len(unique_words))}
    sent_vocab = {"word2id": unique_words_dict, "id2word": unique_words}
    json_object = json.dumps(sent_vocab)
    with open("./vocab/sent_vocab.json", "w") as outfile:
        outfile.write(json_object)

    # Write the unique words into a text file
    with open('./data/data.txt', 'w', encoding='utf-8') as f:
        for word in unique_words:
            f.write(word + " ")

    # Train the fasttext model
    model = fasttext.train_unsupervised('./data/data.txt', model='skipgram', minCount=1, dim=300)
    model.save_model('./data/my_model.bin')

    return unique_words_dict

def pretrained(target_vocab, emb_dim=300):
    # Load pre-trained model
    # model = fasttext.load_model('./data/Pre-trained embeddings/crawl-300d-2M-subword.bin')
    model = fasttext.load_model('./data/my_model.bin')
    print("Done loading the pre-trained model.")

    matrix_len = len(target_vocab)
    weights_matrix = np.zeros((matrix_len, emb_dim))
    words_found = 0

    for word, i in target_vocab.items():
        try:
            weights_matrix[i] = np.array(model[word]).astype(np.float)
            words_found += 1
        except KeyError:
            weights_matrix[i] = np.random.normal(scale=0.6, size=(emb_dim,))
    print("Total number of words are ", len(target_vocab))
    print("Total number of words found in pre-trained embeddings are ", words_found)

    b = weights_matrix.tolist()
    file_path = "./data/weights_matrix.json"
    json.dump(b, codecs.open(file_path, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)

    return weights_matrix

def iobes_iob(tags):
    """
    IOBES -> IOB
    """
    new_tags = []
    for curr_set in tags:
        temp_tags = []
        for j, tag in enumerate(curr_set):
            if tag.split('-')[0] == 'B':
                temp_tags.append(tag)
            elif tag.split('-')[0] == 'I':
                temp_tags.append(tag)
            elif tag.split('-')[0] == 'S':
                temp_tags.append(tag.replace('S-', 'B-'))
            elif tag.split('-')[0] == 'E':
                temp_tags.append(tag.replace('E-', 'I-'))
            elif tag.split('-')[0] == 'O':
                temp_tags.append(tag)
            else:
                temp_tags.append(tag)
                # raise Exception('Invalid format!')
        new_tags.append(temp_tags)
    return new_tags

def main():
    args = docopt(__doc__)
    random.seed(0)
    torch.manual_seed(0)
    if args['--cuda']:
        torch.cuda.manual_seed(0)
    if args['train']:
        unique_tags_ner, unique_tags_entity, unique_words = preprocess_data(args, 'TRAIN')
        unique_words_dict = create_vocab(unique_tags_ner, unique_tags_entity, unique_words)
        print("Done preprocessing the data")
        weights_matrix = pretrained(unique_words_dict)
        print("Done computing the weights matrix")
        train(args, weights_matrix)
    elif args['test']:
        # Load the weights matrix file generated while training
        file_path = "./data/weights_matrix.json"
        obj_text = codecs.open(file_path, 'r', encoding='utf-8').read()
        b_new = json.loads(obj_text)
        weights_matrix = np.array(b_new)

        # Get the unique words and unique tags from the test file
        unique_tags_ner, unique_tags_entity, unique_words = preprocess_data(args, 'TEST')
        # Add the unique words from the test data (not present in train data) to the dictionary
        # Load the train vocab
        with open('./vocab/sent_vocab.json') as json_file:
            train_vocab = json.load(json_file)
        train_words = train_vocab["id2word"]
        model = fasttext.load_model('./data/my_model.bin')
        final_words = list()
        for word in unique_words:
            if word in train_words:
                continue
            else:
                final_words.append(word)
        # If there are new words
        if len(final_words) > 0:
            unique_words_dict = {final_words[i]: i+len(weights_matrix) for i in range(len(final_words))}

            # Update the weights_matrix
            matrix_len = len(unique_words_dict)+len(weights_matrix)
            final_weights_matrix = np.zeros((matrix_len, 300))
            # Rewrite the train weights
            for i in range(len(weights_matrix)):
                final_weights_matrix[i] = weights_matrix[i]
            # Write the test weights
            for word, i in unique_words_dict.items():
                try:
                    final_weights_matrix[i] = np.array(model.get_word_vector(word)).astype(np.float)
                except KeyError:
                    final_weights_matrix[i] = np.random.normal(scale=0.6, size=(300,))

            final_dict = {**unique_words_dict, **train_vocab["word2id"]}
            final_id2word = train_words+final_words
            sent_vocab = {"word2id": final_dict, "id2word": final_id2word}
            json_object = json.dumps(sent_vocab)
            with open("./vocab/sent_vocab.json", "w") as outfile:
                outfile.write(json_object)

            print("Finally here!!")
            b = final_weights_matrix.tolist()
            file_path = "./data/weights_matrix.json"
            json.dump(b, codecs.open(file_path, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)

            test(args, final_weights_matrix)
        else:
            print("It entered here!")
            test(args, weights_matrix)


if __name__ == '__main__':
    main()
