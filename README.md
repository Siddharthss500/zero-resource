```bash

python run.py train MTL ./data/train.txt ./vocab/sent_vocab.json ./vocab/tag_vocab_ner.json ./vocab/tag_vocab_entity.json --cuda --validation-every 100 --max-decay 1 --embed-size 300 --max-epoch 100

python run.py test MTL ./data/tech_test.txt ./vocab/sent_vocab.json ./vocab/tag_vocab_ner.json ./vocab/tag_vocab_entity.json ./model/model.pth --cuda --validation-every 100 --max-decay 1 --embed-size 300 --max-epoch 100

```