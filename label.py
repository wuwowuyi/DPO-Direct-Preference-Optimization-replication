import json
from pathlib import Path
from urllib.parse import urlparse

import httpx
import torch
from transformers import GPT2Tokenizer

# Human labelled data provided by OpenAI
label_url = {
    'sentiment': 'https://openaipublic.blob.core.windows.net/lm-human-preferences/labels/sentiment/offline_5k.json',
    'descriptiveness': 'https://openaipublic.blob.core.windows.net/lm-human-preferences/labels/descriptiveness/offline_5k.json',
    'tldr': 'https://openaipublic.blob.core.windows.net/lm-human-preferences/labels/tldr/online_45k.json',
    'cnndm': 'https://openaipublic.blob.core.windows.net/lm-human-preferences/labels/cnndm/online_45k.json'
}

def parse_url(url):
    result = urlparse(url)
    if result.scheme == 'https':
        assert result.netloc == 'openaipublic.blob.core.windows.net'
        return result.path.lstrip('/')
    else:
        raise Exception(f'Could not parse {url} as an Azure url')


def download_file_cached(task: str, cache_dir: str) -> Path:
    """ Given an Azure path url, caches the contents locally.
        WARNING: only use this function if contents under the path won't change!
        """
    assert task in label_url, f"task {task} is not supported"
    path = parse_url(label_url[task])
    filename = '_'.join(path.rsplit('/')[-2:])  # path is like 'lm-human-preferences/labels/tldr/online_45k.json'
    local_file = Path(cache_dir) / filename
    if not local_file.exists():
        print(f'Downloading training labels {label_url[task]}')
        with httpx.stream('GET', label_url[task]) as r, open(local_file, 'wb') as f:
            for chunk in r.iter_bytes(chunk_size=8192 * 8):
                f.write(chunk)

    return local_file


def download_labels(task: str, labels_dir: str = 'datasets/human_label') -> dict[str, torch.Tensor]:
    """
    Download human labelled data provided by OpenAI and put into a replay buffer.
    """
    with open(download_file_cached(task, labels_dir)) as f:
        results = json.load(f)
        print('Num labels found in source:', len(results))

    # results is a list of items in the best-of-4 format:
    # [{'query':..., 'sample0':..., 'sample1':..., 'sample2':,, 'sample3':.., 'best':..},...]
    # All the queries have the same length.
    # All the samples have the same length too.
    queries, prefers, others = [], [], []
    for d in results:
        queries.append(torch.as_tensor(d['query'], dtype=torch.int32))
        prefers.append(torch.as_tensor(d[f'sample{d["best"]}'], dtype=torch.int32))
        for i in range(4):
            if i == d['best']:
                continue
            others.append(torch.as_tensor(d[f'sample{i}'], dtype=torch.int32))
    return {'query': torch.stack(queries), 'prefer': torch.stack(prefers), 'other': torch.stack(others)}


class LabelBuffer:

    def __init__(self, data: dict[str, torch.Tensor], k: int = 3):
        """
        Constructed from OpenAI's best-of-4 human labelled datasets.

        data - contains 3 keys, `query`, `prefer` and `other`, whose values sorted in order. Each query has one preferred response.
        k - for each query, how many non-preferred responses are in `other`. For best-of-4, k is 3, i.e., one preferred, 3 non-preferred.
            The query for other[i] is query[i//k].
        """
        self.data = data
        self.size = data['other'].shape[0]
        self.k = k

    def __len__(self):
        return self.size

    def get_batch(self, idx: torch.Tensor):
        query_idx, n = idx // self.k, len(idx)
        queries = self.data['query'][query_idx]
        prefers = self.data['prefer'][query_idx]
        other = self.data['other'][idx]

        win = torch.cat((queries, prefers), dim=-1)
        lose = torch.cat((queries, other), dim=-1)
        batch = torch.cat((win, lose))
        win_idx = torch.arange(0, n)  # first half
        lose_idx = torch.arange(n, 2 * n)  # second half
        return batch, win_idx, lose_idx

    def get_eval_query_response(self, k=3):
        return self.data['query'][:k], self.data['prefer'][:k]


def _test_download(task_name):
    labels = download_labels(task_name)
    data_buffer = LabelBuffer(labels)
    indices = torch.randint(100, size=(3,))
    data, prefer, other = data_buffer.get_batch(indices)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    data.masked_fill_(data == 50259, 220)  # 220 is empty space ' '
    for i, d in enumerate(data):
        print(f'\nthe {i}-th query response:')
        print(tokenizer.decode(d))
    assert len(data) == (len(prefer) + len(other))


if __name__ == '__main__':
    _test_download('sentiment')

