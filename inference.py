import torch
from utils import erase, highlight

import json
from tqdm.notebook import tqdm

def demo(iterator, h = None, w = None, connection = 1, tokenizer = None, directory = 'result-cnndm-test.json'):

	if h is not None:
		h.eval()
		device = h.device
	
	if w is not None:
		w.eval()
		device = w.device
	

	with open(directory, 'w') as f:

		with torch.no_grad():

			for i, batch in enumerate(tqdm(iterator)):

				src = batch['article_ids'].to(device)
				trg = batch['highlights_ids'].to(device)
				article_attention_mask = batch['article_attention_mask'].to(device)
				
				connected = torch.rand(1) < connection
				if h is not None:
					preds = torch.cat((h(src[:, :512], attention_mask=article_attention_mask[:, :512]).logits, h(src[:, 512:], attention_mask=article_attention_mask[:, 512:]).logits), dim = 1)
					
					if connected:
						src_erased = erase(src, torch.logical_and(preds.argmax(2), article_attention_mask)).to(device)
				else:
					connected = False
				
				if not connected:
					print('The model is not highlighting.')
					src_erased = erase(src, highlight(src, trg)).to(device)

				if w is not None:
					summary_ids = w.generate(src_erased, num_beams=4, max_length=1024, early_stopping=True)
					predictions = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids]
				else:
					print('The model is not writing.')
					predictions = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in src_erased]
				references = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in trg]

				data = [json.dumps({'reference':references[i], 'prediction':predictions[i]}) + '\n' for i in range(len(predictions))]
				f.writelines(data)

	with open(directory, 'r') as f:
		data = [json.loads(x) for x in f.readlines()]
		
	return data
