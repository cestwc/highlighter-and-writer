import torch

def highlight(article_tensor, highlights_tensor):
	unique_tensor = torch.unique(highlights_tensor, dim = 1)
	masks = []
	for i in range(unique_tensor.shape[1]):
		masks.append(torch.eq(article_tensor, unique_tensor[:,i].unsqueeze(1).repeat(1, article_tensor.shape[1])))
	return torch.logical_and(sum(masks).bool(), article_tensor > 44).long()

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
