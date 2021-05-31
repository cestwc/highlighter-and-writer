import torch
from utils import erase

import json
from tqdm.notebook import tqdm

def demo(iterator, h = None, w = None, tokenizer = None, directory = 'result-cnndm-test.json'):

	h.eval()
	w.eval()
	device = h.device

	with open(directory, 'w') as f:

		with torch.no_grad():

			for i, batch in enumerate(tqdm(iterator)):

				src = batch['article_ids'].to(device)
				trg = batch['highlights_ids'].to(device)
				article_attention_mask = batch['article_attention_mask'].to(device)

				preds = torch.cat((h(src[:, :512], attention_mask=article_attention_mask[:, :512]).logits, h(src[:, 512:], attention_mask=article_attention_mask[:, 512:]).logits), dim = 1)

				src_erased = erase(src, torch.logical_and(preds.argmax(2), article_attention_mask)).to(device)
				summary_ids = w.generate(src_erased, num_beams=4, max_length=1024, early_stopping=True)

				predictions = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids]
				references = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in trg]

				data = [json.dumps({'reference':references[i], 'prediction':predictions[i]}) + '\n' for i in range(len(predictions))]
				f.writelines(data)

	with open(directory, 'r') as f:
		data = [json.loads(x) for x in f.readlines()]
		
	return data
