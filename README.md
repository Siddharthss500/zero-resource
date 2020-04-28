# Zero Resource Cross-Domain Named Entity Recognition

This is a Pytorch implementation of BiLSTM-CRF+MTL+MoEE for Cross Domain Named Entity Recognition, which is described in https://arxiv.org/pdf/2002.05923.pdf.

### Data

The corpus is found in the data folder. The model uses CoNLL 2003 data for training and validation. The test data is the CBS SciTech News Dataset taken from https://github.com/jiachenwestlake/Cross-Domain_NER. 

### How to run the code

```python

python run.py train METHOD TRAIN SENT_VOCAB TAG_VOCAB_NER TAG_VOCAB_ENTITY [options]

python run.py test METHOD TEST SENT_VOCAB TAG_VOCAB_NER TAG_VOCAB_ENTITY MODEL [options]

```

For example,

```python

python run.py train MTL ./data/train.txt ./vocab/sent_vocab.json ./vocab/tag_vocab_ner.json ./vocab/tag_vocab_entity.json --cuda --validation-every 100 --max-decay 1 --embed-size 300 --max-epoch 100

python run.py test MTL ./data/tech_test.txt ./vocab/sent_vocab.json ./vocab/tag_vocab_ner.json ./vocab/tag_vocab_entity.json ./model/model.pth --cuda --validation-every 100 --max-decay 1 --embed-size 300 --max-epoch 100

```

### Reference

1. https://github.com/Gxzzz/BiLSTM-CRF
2. https://github.com/jiachenwestlake/Cross-Domain_NER