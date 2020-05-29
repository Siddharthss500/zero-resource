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

### Results

| | Author's Results || Our Results ||||
| Model 					   | FastText-Pretrained | | FastText-Fine tuned | | FastText-Pretrained | |
| | unfreeze | freeze | unfreeze | freeze | unfreeze | freeze |
| ------------- |:------------:| -----:|----: | ------------- |:------------:| -----:|
| BiLSTM-CRF 	 			   | 63.18 | 67.89 | 63.38 | 67.64 | 64.41 | 67.56 |
| BiLSTM-CRF w/ MTL            | 64.62 | 69.58 | 66.5  | 64.7  | 68.84 | 68.89 |
| BiLSTM-CRF w/ MTL (Separate) | - 	   | - 	   | 66.3  | 66.1  | 68.84 | 68.89 |
| BiLSTM-CRF w/ MoEE 		   | 65.24 | 69.25 | 62.3  | 63.19 | 68.7  | 67.94 |
| BiLSTM-CRF w/ MTL and MoEE   | 64.88 | 70.04 | 49.98 | 65.48 | 67.24 | 68.33 |
| Mod1 						   | - 	   | -     | 66.27 | 65.57 | 68.31 | 70.37 |
| Mod2 						   | -     | -     | 45.66 | 61.91 | 65.33 | 69.36 |		


### Contributors

1. [Siddharth Sundararajan](https://github.com/Siddharthss500)
2. [Nitin Karolla Reddy](https://github.com/nitinkarolla)

### Reference

1. https://github.com/Gxzzz/BiLSTM-CRF
2. https://github.com/jiachenwestlake/Cross-Domain_NER