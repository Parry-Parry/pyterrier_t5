import torch
import pandas as pd
import numpy as np
import re
from transformers import T5Tokenizer
from pyterrier_t5.modeling_fid import FiD
import pyterrier as pt
from tqdm.auto import tqdm

def _iter_windows(n, window_size, stride):
    # TODO: validate window_size and stride
    for start_idx in tqdm(range((n // stride) * stride, -1, -stride), unit='window'):
        end_idx = start_idx + window_size
        if end_idx > n:
            end_idx = n
        window_len = end_idx - start_idx
        if start_idx == 0 or window_len > stride:
            yield start_idx, end_idx, window_len

class LiT5(pt.Transformer):
    def __init__(self, model_path='castorini/LiT5-Distill-large', batch_size=16, verbose=True, bfloat16=None, window_size=20, stride=10, passes=1):
        self.tokenizer = T5Tokenizer.from_pretrained(model_path, return_dict=False, legacy=False, use_fast=True)
        self.model = FiD.from_pretrained(model_path, from_flax=False).cuda().eval()
        self.model.encoder.config.n_passages = window_size
        self.model.encoder.config.batch_size = batch_size
        if bfloat16 is None:
            try:
                self.model = self.model.bfloat16()
                bfloat16 = True
            except:
                bfloat16 = False
        elif bfloat16:
            self.model = self.model.bfloat16()
        self.bfloat16 = bfloat16
        self.passes = passes
        self.window_size = window_size
        self.stride = stride

    def transform(self, inp):
        res = {
            'qid': [],
            'query': [],
            'docno': [],
            'text': [],
            'rank': [],
        }
        with torch.no_grad():
            for (qid, query), query_results in tqdm(inp.groupby(['qid', 'query']), unit='q'):
                query_results = query_results.sort_values('score', ascending=False)
                doc_ids = query_results['docno'].to_numpy()
                doc_texts = query_results['text'].to_numpy()
                for p in range(self.passes):
                    for start_idx, end_idx, window_len in _iter_windows(len(query_results), self.window_size, self.stride):
                        template = "Search Query: {q} Passage: [{i}] {d} Relevance Ranking: "
                        passages = [template.format(q=query, i=i+1, d=text) for i, text in enumerate(doc_texts[start_idx:end_idx].tolist() + ["" for _ in range(end_idx - start_idx, self.window_size)])]
                        inputs = self.tokenizer.batch_encode_plus(passages, return_tensors="pt", padding='max_length', max_length=150, truncation=True)
                        outputs = self.model.generate(
                            input_ids=inputs['input_ids'].cuda().reshape(1, -1),
                            attention_mask=inputs['attention_mask'].cuda().reshape(1, -1),
                            max_length=100,
                            do_sample=False,
                        )
                        output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                        output = re.sub(r'[^0-9]', ' ', output) # clean outputs (keep only digits)
                        output = [int(x)-1 for x in output.split()] # convert to integer
                        output = list({x: 0 for x in output if 0 <= x < window_len}.keys()) # remove duplicates (but keep order) and remove anything out of range
                        output = output + [i for i in range(window_len) if i not in output] # backfill missing passages
                        new_idxs = start_idx + np.array(output)
                        orig_idxs = np.arange(start_idx, end_idx)
                        doc_ids[orig_idxs] = doc_ids[new_idxs]
                        doc_texts[orig_idxs] = doc_texts[new_idxs]
                res['qid'].extend([qid] * len(doc_ids))
                res['query'].extend([query] * len(doc_ids))
                res['docno'].extend(doc_ids)
                res['text'].extend(doc_texts)
                res['rank'].extend(list(range(len(doc_ids))))
        res = pd.DataFrame(res)
        res['score'] = -res['rank'].astype(float)
        return res
