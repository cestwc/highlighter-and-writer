import os
import math
import time

def metric_line(metric, mode = 'train'):
	out = ['\t' + mode.upper()]
	if 'h' in metric:
		out.append(f'HGLR Loss: {metric["h"]:.3f}')
		for i in range(4):
			out.append(['recall', 'prec', 'f1', 'f2'][i] + f': {metric["metric"][i]*100:.2f}%')
	if 'w' in metric:
		out.append(f'WRTR Loss: {metric["w"]:.3f}')
		out.append(f'PPL: {math.exp(metric["w"]):7.3f}')
	return ' | '.join(out)


def save_best_models(metric, best_valid_loss, folder, h = None, w = None):
	if 'h' in metric and h is not None:
		if metric['h'] < best_valid_loss['h']:
			best_valid_loss['h'] = metric['h']
			torch.save(h.state_dict(), os.path.join(folder + 'hglr-model.pt'))
	if 'w' in metric and w is not None:
		if metric['w'] < best_valid_loss['w']:
			best_valid_loss['w'] = metric['w']
			torch.save(w.state_dict(), os.path.join(folder + 'wrtr-model.pt'))
	return best_valid_loss

# from tutorial
def epoch_time(start_time, end_time):
	elapsed_time = end_time - start_time
	elapsed_mins = int(elapsed_time / 60)
	elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
	return elapsed_mins, elapsed_secs
