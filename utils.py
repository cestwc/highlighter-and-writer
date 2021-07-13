import re
import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
pattern = re.compile(r'\b(' + r'|'.join(stopwords.words('english')) + r')\b\s*' + r'|[^\w\s]')

removeStopWords = lambda x:pattern.sub('', x)

def inOtherList(a, b):
	b = set(b)    
	return list(map(lambda x: x in b, a))

import torch

def erase(article_tensor, highlight_mask):
	highlight_mask = highlight_mask.bool()
	l = [torch.masked_select(article_tensor[i], highlight_mask[i]) for i in range(len(article_tensor))]
	return torch.nn.utils.rnn.pad_sequence(l, batch_first=True, padding_value=1).long()

def binary_metric(preds, y):
	rounded_preds = preds.argmax(2)
	t = rounded_preds == y
	p = rounded_preds == 1
	tp = torch.sum(torch.logical_and(t, p).float(), dim = 1)
	fp = torch.sum(torch.logical_and(torch.logical_not(t), p).float(), dim = 1)
	fn = torch.sum(torch.logical_and(torch.logical_not(t), torch.logical_not(p)).float(), dim = 1)

	recall = tp / (tp + fn + 1e-10)
	prec = tp / (tp + fp + 1e-10)
	f1 = 2 * tp / (2 * tp + fp + fn + 1e-10)
	f2 = 5 * tp / (5 * tp + 4 * fn + fp + 1e-10)
	return torch.tensor([recall.mean(), prec.mean(), f1.mean(), f2.mean()])

def labelcounts(iterator, n_classes = 2):
	counts = torch.ones(n_classes) * 2 # normally, it should start from zero, though we start with 2 to avoid 'divide by zero'
	for batch in iterator:
		h_mask = torch.logical_and(batch['highlight_mask'], batch['article_attention_mask']).long()
		for i in range(n_classes):
			counts[i] += torch.sum(h_mask == i)
	return counts
