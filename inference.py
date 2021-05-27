import torch
from utils import highlight, erase

def demo(src, trg, highlighter, writer, tokenizer):
	if isinstance(src, str):
		src = tokenizer(src, max_length=1024, return_tensors="pt")
		trg = tokenizer(trg, max_length=1024, return_tensors="pt")
		
	src = src.to(writer.device)
	trg = trg.to(writer.device)
	
	h_mask = highlight(src, trg)
	
	outputs = highlighter(**inputs, labels=h_mask)
	logits = outputs.logits
	
	src_erased = erase(src, logits.argmax(2)).to(writer.device)
	summary_ids = writer.generate(src_erased, num_beams=4, max_length=100, early_stopping=True)
	
	return [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids]
