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

<table> 
	<tr> 
		<th rowspan="3">Model</th>
		<th colspan="2">Author's Results</th> 
		<th colspan="4">Our Results</th> 
	</tr> 
	<tr> 
		<th colspan="2">FastText-Pretrained</th>
		<th colspan="2">FastText-Fine tuned</th>
		<th colspan="2">FastText-Pretrained</th>
	</tr> 
	<tr>
		<th>unfreeze</th>
		<th>freeze</th>
		<th>unfreeze</th>
		<th>freeze</th>
		<th>unfreeze</th>
		<th>freeze</th>
	</tr>
	<tr>
		<td>BiLSTM-CRF</td>
		<td>63.18</td>
		<td>67.89</td>
		<td>63.38</td>
		<td>67.64</td>
		<td>64.41</td>
		<td>67.56</td>
	</tr>
	<tr>
		<td>BiLSTM-CRF w/ MTL</td>
		<td>64.62</td>
		<td>69.58</td>
		<td>66.5</td>
		<td>64.7</td>
		<td>68.84</td>
		<td>68.89</td>
	</tr>
	<tr>
		<td>BiLSTM-CRF w/ MTL (Separate)</td>
		<td>-</td>
		<td>-</td>
		<td>66.3</td>
		<td>66.1</td>
		<td>68.84</td>
		<td>68.89</td>
	</tr>
	<tr>
		<td>BiLSTM-CRF w/ MoEE</td>
		<th>65.24</th>
		<td>69.25</td>
		<td>62.3</td>
		<td>63.19</td>
		<td>68.7</td>
		<td>67.94</td>
	</tr>
	<tr>
		<td>BiLSTM-CRF w/ MTL and MoEE</td>
		<td>64.88</td>
		<th>70.04</th>
		<td>49.98</td>
		<td>65.38</td>
		<td>67.24</td>
		<td>68.33</td>
	</tr>
	<tr>
		<td>Mod1</td>
		<td>-</td>
		<td>-</td>
		<td>66.27</td>
		<td>65.57</td>
		<th>68.31</th>
		<th>70.37</th>
	</tr>
	<tr>
		<td>Mod2</td>
		<td>-</td>
		<td>-</td>
		<td>45.66</td>
		<td>61.91</td>
		<td>65.33</td>
		<th>69.36</th>
	</tr>
</table>

		


### Contributors

1. [Siddharth Sundararajan](https://github.com/Siddharthss500)
2. [Nitin Reddy Karolla](https://github.com/nitinkarolla)

### Reference

1. https://github.com/Gxzzz/BiLSTM-CRF
2. https://github.com/jiachenwestlake/Cross-Domain_NER