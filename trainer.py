import torch
from tqdm.notebook import tqdm

from utils import highlight, erase, binary_metric

def tokenClassificationTrainStep(model, optimizer, clip, criterion, src, labels, attention_mask = None):

	optimizer.zero_grad()

	logits = torch.cat((model(src[:, :512]).logits, model(src[:, 512:]).logits), dim = 1)

	if attention_mask is not None:
		active_loss = attention_mask.view(-1) == 1
		active_logits = logits.view(-1, model.num_labels)
		active_labels = torch.where(
			active_loss, labels.view(-1), torch.tensor(criterion.ignore_index).type_as(labels)
		)
		loss = criterion(active_logits, active_labels)
	else:
		loss = criterion(logits.view(-1, model.num_labels), labels.view(-1))

	#logits = [batch size, src len]

	score = binary_metric(logits, labels)

	loss.backward()

	torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

	optimizer.step()

	return {'loss':loss.item(), 'metric':score}, logits

def tokenClassificationEvalStep(model, criterion, src, labels, attention_mask = None):

	logits = torch.cat((model(src[:, :512]).logits, model(src[:, 512:]).logits), dim = 1)
	
	if attention_mask is not None:
		active_loss = attention_mask.view(-1) == 1
		active_logits = logits.view(-1, model.num_labels)
		active_labels = torch.where(
			active_loss, labels.view(-1), torch.tensor(criterion.ignore_index).type_as(labels)
		)
		loss = criterion(active_logits, active_labels)
	else:
		loss = criterion(logits.view(-1, model.num_labels), labels.view(-1))

	score = binary_metric(logits, labels)

	return {'loss':loss.item(), 'metric':score}, logits

def conditionalGenerationTrainStep(model, optimizer, clip, criterion, src, trg):
	optimizer.zero_grad()

	output = model(src, decoder_input_ids = trg[:,:-1])[0]

	#output = [batch size, trg len - 1, output dim]
	#trg = [batch size, trg len]

	output_dim = output.shape[-1]

	output = output.contiguous().view(-1, output_dim)
	trg = trg[:,1:].contiguous().view(-1)

	# print(src.shape, output.shape, trg.shape)
	# print(trg)

	#output = [batch size * trg len - 1, output dim]
	#trg = [batch size * trg len - 1]

	loss = criterion(output, trg)

	loss.backward()

	torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

	optimizer.step()

	return {'loss':loss.item()}

def conditionalGenerationEvalStep(model, criterion, src, trg):
	output = model(src, decoder_input_ids = trg[:,:-1])[0]

	#output = [batch size, trg len - 1, output dim]
	#trg = [batch size, trg len]

	output_dim = output.shape[-1]

	output = output.contiguous().view(-1, output_dim)
	trg = trg[:,1:].contiguous().view(-1)

	#output = [batch size * trg len - 1, output dim]
	#trg = [batch size * trg len - 1]

	loss = criterion(output, trg)

	return {'loss':loss.item()}

def train(iterator, clip, h = None, optH = None, criterionH = None, w = None, optW = None, criterionW = None, connection = 0.5, tuning = False):

	epoch_loss = {}
	if h is not None:
		if not tuning:
			h.train()
		else:
			h.eval()
		epoch_loss['h'] = 0
		epoch_loss['metric'] = torch.zeros(4)
		device = h.device
	if w is not None:
		w.train()
		epoch_loss['w'] = 0
		device = w.device

	for _, batch in enumerate(tqdm(iterator)):

		src = batch['article_ids'].to(device)
		trg = batch['highlights_ids'].to(device)
		article_attention_mask = batch['article_attention_mask'].to(device)
		h_mask = highlight(src, trg)
		if 'h' in epoch_loss:
			if not tuning:
				outputs, preds = tokenClassificationTrainStep(h, optH, clip, criterionH, src, h_mask, article_attention_mask)
			else:
				with torch.no_grad():
					outputs, preds = tokenClassificationEvalStep(h, criterionH, src, h_mask, article_attention_mask)
			epoch_loss['h'] += outputs['loss']
			epoch_loss['metric'] += outputs['metric']

		if 'w' in epoch_loss:
			src_erased = erase(src, torch.logical_and(preds.argmax(2), article_attention_mask)).to(device) if 'h' in epoch_loss and torch.rand(1) < connection else erase(src, h_mask).to(device)
			outputs = conditionalGenerationTrainStep(w, optW, clip, criterionW, src_erased, trg)
			epoch_loss['w'] += outputs['loss']

	return {key:value/len(iterator) for key, value in epoch_loss.items()}

def evaluate(iterator, h = None, criterionH = None, w = None, criterionW = None, connection = 0.5):

	epoch_loss = {}
	if h is not None:
		h.eval()
		epoch_loss['h'] = 0
		epoch_loss['metric'] = torch.zeros(4)
		device = h.device
	if w is not None:
		w.eval()
		epoch_loss['w'] = 0
		device = w.device

	with torch.no_grad():

		for _, batch in enumerate(tqdm(iterator)):

			src = batch['article_ids'].to(device)
			trg = batch['highlights_ids'].to(device)
			article_attention_mask = batch['article_attention_mask'].to(device)
			h_mask = highlight(src, trg)
			if 'h' in epoch_loss:
				outputs, preds = tokenClassificationEvalStep(h, criterionH, src, h_mask, article_attention_mask)
				epoch_loss['h'] += outputs['loss']
				epoch_loss['metric'] += outputs['metric']

			if 'w' in epoch_loss:
				src_erased = erase(src, torch.logical_and(preds.argmax(2), article_attention_mask)).to(device) if 'h' in epoch_loss and torch.rand(1) < connection else erase(src, h_mask).to(device)
				outputs = conditionalGenerationEvalStep(w, criterionW, src_erased, trg)
				epoch_loss['w'] += outputs['loss']


	return {key:value/len(iterator) for key, value in epoch_loss.items()}
