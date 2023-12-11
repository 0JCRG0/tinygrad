#!/usr/bin/env python3
from __future__ import annotations
from tinygrad import Tensor, nn
import argparse
import os
import json
import sys
import torch
#from torch import stack
from pathlib import Path
from typing import Any
from torch import load
import numpy as np
from tinygrad.helpers import Timing, colored, getenv, fetch
from gguf import GGUFReader, GGUFValueType  # noqa: E402

"""
I have loaded all the tensors from the GGUF file. There's a total of 
219 tensors. Each tensor is a ReaderTensor() class, which is defined as follows:---

https://github.com/ggerganov/llama.cpp/blob/5a7d3125e7c24f223659b7f0b7aa7736986e92c0/gguf-py/gguf/gguf_reader.py#L53


class ReaderTensor(NamedTuple):
    name: str
    tensor_type: GGMLQuantizationType
    shape: npt.NDArray[np.uint32]
    n_elements: int
    n_bytes: int
    data_offset: int
    data: npt.NDArray[Any]
    field: ReaderField

    ----

    def safe_load_metadata(fn:Union[Tensor,str]) -> Tuple[Tensor, int, Any]:
        t = fn if isinstance(fn, Tensor) else Tensor.empty(os.stat(fn).st_size, dtype=dtypes.uint8, device=f"disk:{fn}")
        json_len = t[0:1].cast(dtypes.int64).numpy()[0]
        return (t, json_len, json.loads(t[8:8+json_len].numpy().tobytes()))

        def safe_load(fn:Union[Tensor,str]) -> Dict[str, Tensor]:
        t, json_len, metadata = safe_load_metadata(fn)
        return {k:t[8+json_len+v['data_offsets'][0]:].cast(safe_dtypes[v['dtype']])[:prod(v['shape'])].reshape(v['shape']) for k,v in metadata.items() if k != "__metadata__"}

"""

def reader_tensor_to_dict(reader_tensor):
    return {
        reader_tensor.name: {
            'data_offsets': [reader_tensor.data_offset],
            'dtype': str(reader_tensor.data.dtype),
            'shape': reader_tensor.shape.tolist()
        }
    }


weigths = 'weights/llama-2-7b.Q4_0.gguf'

reader = GGUFReader(weigths, 'r')
#print(f'\n* Dumping {len(reader.tensors)} tensor(s)')
TENSORS = reader.tensors

#TODO IDEA: Combining the tensors into 1?
#combined_tensor = Tensor.stack(TENSORS, dim=0)
#print(combined_tensor, type(combined_tensor))

print(TENSORS[0], type(TENSORS))

exit(0)



if __name__ == "__main__":

    with Timing("download weights: "):
        # Convert the numpy array into a Tensor
        part_1 = nn.state.torch_load(TENSORS)
        print(part_1)

        

exit(0)



tensor_dicts = [reader_tensor_to_dict(tensor) for tensor in TENSORS]

with open('loading_gguf.txt', 'w') as file:
    file.write(f"tensor_dicts: \n\n{tensor_dicts}\n\n")


metadata = {}
for tensor_dict in tensor_dicts:
    metadata.update(tensor_dict)

with open('loading_gguf.txt', 'a') as file:
    file.write(f"metadata: \n\n{metadata} type: {type(metadata)}")

with open('weights/llama-2-7b.Q4_0-metadata.json', 'w') as f:
    json.dump(tensor_dicts, f)


exit(0)


MAX_CONTEXT = getenv("MAX_CONTEXT", 4096)

MODEL_PARAMS = {
"1": {
    "7B": {
    "args": {"dim": 4096, "n_heads": 32, "n_layers": 32, "norm_eps": 1e-06, "vocab_size": 32000, "hidden_dim": 11008},
    "files": 1,
    },
    "13B": {
    "args": {"dim": 5120, "n_heads": 40, "n_layers": 40, "norm_eps": 1e-06, "vocab_size": 32000, "hidden_dim": 13824},
    "files": 2,
    },
    "30B": {
    "args": {"dim": 6656, "n_heads": 52, "n_layers": 60, "norm_eps": 1e-06, "vocab_size": 32000, "hidden_dim": 17920},
    "files": 4,
    },
    "65B": {
    "args": {"dim": 8192, "n_heads": 64, "n_layers": 80, "norm_eps": 1e-05, "vocab_size": 32000, "hidden_dim": 22016},
    "files": 8,
    },
},
"2": {
    "7B": {
    "args": {"dim": 4096, "n_heads": 32, "n_layers": 32, "norm_eps": 1e-05, "vocab_size": 32000, "hidden_dim": 11008},
    "files": 1,
    },
    "13B": {
    "args": {"dim": 5120, "n_heads": 40, "n_layers": 40, "norm_eps": 1e-05, "vocab_size": 32000, "hidden_dim": 13824},
    "files": 2,
    },
    "70B": {
    "args": {"dim": 8192, "n_heads": 64, "n_kv_heads": 8, "n_layers": 80, "norm_eps": 1e-05, "vocab_size": 32000, "hidden_dim": 28672},
    "files": 8,
    },
},
"code": {
    "7B": {
    "args": {"dim": 4096, "n_layers": 32, "n_heads": 32, "norm_eps": 1e-05, "rope_theta": 1000000, "vocab_size": 32016, "hidden_dim": 11008},
    "files": 1,
    },
    "7B-Python": {
    "args": {"dim": 4096, "n_layers": 32, "n_heads": 32, "norm_eps": 1e-05, "rope_theta": 1000000, "vocab_size": 32000, "hidden_dim": 11008},
    "files": 1,
    },
    "7B-Instruct": {
    "args": {"dim": 4096, "n_layers": 32, "n_heads": 32, "norm_eps": 1e-05, "rope_theta": 1000000, "vocab_size": 32016, "hidden_dim": 11008},
    "files": 1,
    },
    "13B": {
    "args": {"dim": 5120, "n_layers": 40, "n_heads": 40, "norm_eps": 1e-05, "rope_theta": 1000000, "vocab_size": 32016, "hidden_dim": 13824},
    "files": 2,
    },
    "13B-Python": {
    "args": {"dim": 5120, "n_layers": 40, "n_heads": 40, "norm_eps": 1e-05, "rope_theta": 1000000, "vocab_size": 32000, "hidden_dim": 13824},
    "files": 2,
    },
    "13B-Instruct": {
    "args": {"dim": 5120, "n_layers": 40, "n_heads": 40, "norm_eps": 1e-05, "rope_theta": 1000000, "vocab_size": 32016, "hidden_dim": 13824},
    "files": 2,
    },
    "34B": {
    "args": {"dim": 8192, "n_layers": 48, "n_heads": 64, "n_kv_heads": 8, "norm_eps": 1e-05, "rope_theta": 1000000, "vocab_size": 32000, "hidden_dim": 22016},
    "files": 4,
    },
    "34B-Python": {
    "args": {"dim": 8192, "n_layers": 48, "n_heads": 64, "n_kv_heads": 8, "norm_eps": 1e-05, "rope_theta": 1000000, "vocab_size": 32000, "hidden_dim": 22016},
    "files": 4,
    },
    "34B-Instruct": {
    "args": {"dim": 8192, "n_layers": 48, "n_heads": 64, "n_kv_heads": 8, "norm_eps": 1e-05, "rope_theta": 1000000, "vocab_size": 32000, "hidden_dim": 22016},
    "files": 4,
    },
},
"tiny": {
    "1B": {
    "args": {"dim": 2048, "n_layers": 22, "n_heads": 32, "n_kv_heads": 4, "norm_eps": 1e-05, "vocab_size": 32000, "hidden_dim": 5632},
    "files": 1,
    }
}
}

def is_gguf(path: Union[str, Path]) -> bool:
    path = str(Path(path).resolve())
    with open(path, "rb") as f:
        magic = f.read(2000)
        print(magic)
    #return magic == "GGUF".encode()

is_gguf("weights/llama-2-7b.Q4_0.gguf")


weights = "weights/llama-2-7b.Q4_0.gguf"


with Timing("create model: "):
    model = Transformer(dim=4096, hidden_dim=11008, n_heads=32, n_layers=32, norm_eps=1e-05, vocab_size=32000, linear=AbsmaxQuantizedLinear, max_context=MAX_CONTEXT)
    print(model)


with Timing("weights -> model: "):
    nn.state.load_state_dict(model, convert_from_huggingface(weights=weights, model=model, n_heads=32, n_kv_heads=0), strict=False)