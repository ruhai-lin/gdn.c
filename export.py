"""
This script has functions and utilities for model export.
Export a trained GDN checkpoint to .bin files to be read from and inferenced in C.

Among the "output" formats of .bin files:
- v1: full-precision weights for run.c
- v2: Q8_0 quantized weights for runq.c

Both formats use the same "GDNe" magic and an explicit version in the header,
following llama2.c's export.py/runq.c layout.
"""
import argparse
import os
import struct

import torch

from model import GDNLM, ModelArgs


GDNE_MAGIC = 0x47444E65  # "GDNe" little-endian
HEADER_SIZE = 256

# -----------------------------------------------------------------------------
# common utilities

def serialize_fp32(file, tensor):
    """ writes one fp32 tensor to file that is open in wb mode """
    d = tensor.detach().cpu().view(-1).to(torch.float32).numpy()
    file.write(d.tobytes())


def serialize_int8(file, tensor):
    """ writes one int8 tensor to file that is open in wb mode """
    d = tensor.detach().cpu().view(-1).to(torch.int8).numpy()
    file.write(d.tobytes())


def quantize_q80(w, group_size):
    """
    takes a 2D weight matrix and returns the Q8_0 quantized version
    i.e. symmetric quantization into int8, range [-127,127],
    grouped along each matrix row (layout expected by runq.c)
    """
    assert w.ndim == 2
    rows, cols = w.shape
    assert cols % group_size == 0, f"input dim {cols} not divisible by group_size {group_size}"
    w = w.detach().cpu().float()
    w = w.reshape(rows, cols // group_size, group_size)
    # find the max in each group
    wmax = torch.abs(w).amax(dim=-1)
    # calculate the scaling factor such that float = quant * scale
    scale = wmax / 127.0
    scale = torch.where(wmax == 0, torch.ones_like(scale), scale)  # avoid div-by-zero
    # scale into range [-127, 127]
    quant = w / scale.unsqueeze(-1)
    # round to nearest integer
    int8val = torch.round(quant).clamp_(-127, 127).to(torch.int8)
    # dequantize by rescaling and measure max error
    fp32val = int8val.float() * scale.unsqueeze(-1)
    err = torch.abs(fp32val - w).amax(dim=-1)
    maxerr = err.max().item()
    return int8val.reshape(rows, cols).contiguous(), scale.contiguous(), maxerr

# -----------------------------------------------------------------------------
# version 1: fp32 export (consumed by run.c)

def version1_export(model, filepath):
    """Export GDNLM weights to a C-readable FP32 binary file for run.c."""
    p = model.params
    shared_classifier = torch.equal(model.tok_embeddings.weight, model.output.weight)

    out_file = open(filepath, "wb")
    # first write out the header. the header will be 256 bytes
    # 1) write magic, which will be uint32 of "GDNe" in ASCII
    out_file.write(struct.pack("<I", GDNE_MAGIC))
    # 2) write the format version
    out_file.write(struct.pack("<i", 1))
    # 3) write the params
    out_file.write(
        struct.pack(
            "<11i",
            p.dim,
            p.hidden_dim,
            p.n_layers,
            p.num_heads,
            p.head_k_dim,
            p.head_v_dim,
            p.conv_size,
            p.vocab_size,
            p.max_seq_len,
            int(shared_classifier),
            0,
        )
    )
    pad = HEADER_SIZE - out_file.tell()
    assert pad >= 0
    out_file.write(b"\0" * pad)

    # now write the model: embedding, then each layer's mixer+ffn, final norm
    serialize_fp32(out_file, model.tok_embeddings.weight)
    for layer in model.layers:
        m = layer.mixer
        f = layer.ffn
        serialize_fp32(out_file, layer.attn_norm.weight)
        serialize_fp32(out_file, m.q_proj.weight)
        serialize_fp32(out_file, m.k_proj.weight)
        serialize_fp32(out_file, m.v_proj.weight)
        serialize_fp32(out_file, m.a_proj.weight)
        serialize_fp32(out_file, m.b_proj.weight)
        serialize_fp32(out_file, m.g_proj.weight)
        serialize_fp32(out_file, m.q_conv1d.weight)
        serialize_fp32(out_file, m.k_conv1d.weight)
        serialize_fp32(out_file, m.v_conv1d.weight)
        serialize_fp32(out_file, -torch.exp(m.A_log.float()))
        serialize_fp32(out_file, m.dt_bias)
        serialize_fp32(out_file, m.o_norm.weight)
        serialize_fp32(out_file, m.o_proj.weight)
        serialize_fp32(out_file, layer.ffn_norm.weight)
        serialize_fp32(out_file, f.gate_proj.weight)
        serialize_fp32(out_file, f.down_proj.weight)
        serialize_fp32(out_file, f.up_proj.weight)
    serialize_fp32(out_file, model.norm.weight)
    if not shared_classifier:
        serialize_fp32(out_file, model.output.weight)

    out_file.close()
    print(f"wrote {filepath}")

# -----------------------------------------------------------------------------
# version 2: int8 / Q8_0 export (consumed by runq.c)

def version2_export(model, filepath, group_size=32):
    """
    Export the model weights in Q8_0 into .bin file to be read from C.
    That is:
    - quantize large projection matrices to symmetric int8, in range [-127, 127]
    - norms / a,b proj / conv / A / dt_bias / o_norm stay fp32
    - quantization is done in groups of group_size along each matrix row
    - payload is layer-contiguous: [mixer][ffn] per layer (same order as fp32)
    """
    p = model.params
    shared_classifier = torch.equal(model.tok_embeddings.weight, model.output.weight)

    # validate all Q8 matrices up front
    weights = [
        ("tok_embeddings/output", model.tok_embeddings.weight),
    ]
    for i, layer in enumerate(model.layers):
        m, f = layer.mixer, layer.ffn
        weights.extend([
            (f"layers.{i}.mixer.q_proj", m.q_proj.weight),
            (f"layers.{i}.mixer.k_proj", m.k_proj.weight),
            (f"layers.{i}.mixer.v_proj", m.v_proj.weight),
            (f"layers.{i}.mixer.g_proj", m.g_proj.weight),
            (f"layers.{i}.mixer.o_proj", m.o_proj.weight),
            (f"layers.{i}.ffn.gate_proj", f.gate_proj.weight),
            (f"layers.{i}.ffn.down_proj", f.down_proj.weight),
            (f"layers.{i}.ffn.up_proj", f.up_proj.weight),
        ])
    if not shared_classifier:
        weights.append(("output", model.output.weight))
    for _, w in weights:
        assert w.ndim == 2 and w.shape[1] % group_size == 0, (
            f"weight shape {tuple(w.shape)} not compatible with group_size {group_size}"
        )

    out_file = open(filepath, "wb")
    # first write out the header. the header will be 256 bytes
    # 1) write the shared GDN magic and v2 format version
    out_file.write(struct.pack("<I", GDNE_MAGIC))
    out_file.write(struct.pack("<i", 2))
    # 2) write the params + shared flag + pad (11 ints)
    header = struct.pack(
        "<11i",
        p.dim,
        p.hidden_dim,
        p.n_layers,
        p.num_heads,
        p.head_k_dim,
        p.head_v_dim,
        p.conv_size,
        p.vocab_size,
        p.max_seq_len,
        int(shared_classifier),
        0,
    )
    out_file.write(header)
    # 3) write the group size used for Q8_0 quantization
    out_file.write(struct.pack("<i", group_size))
    pad = HEADER_SIZE - out_file.tell()
    assert pad >= 0
    out_file.write(b"\0" * pad)

    # now write the model: embedding, then each layer's mixer+ffn, final norm
    ew = []

    def write_q8(name, w):
        q, s, err = quantize_q80(w, group_size)
        serialize_int8(out_file, q)
        serialize_fp32(out_file, s)
        ew.append((err, w.shape))
        print(f"{len(ew)}/{len(weights)} quantized {tuple(w.shape)} to Q8_0 with max error {err}")

    write_q8("tok_embeddings/output", model.tok_embeddings.weight)

    for i, layer in enumerate(model.layers):
        m, f = layer.mixer, layer.ffn
        serialize_fp32(out_file, layer.attn_norm.weight)
        write_q8(f"layers.{i}.mixer.q_proj", m.q_proj.weight)
        write_q8(f"layers.{i}.mixer.k_proj", m.k_proj.weight)
        write_q8(f"layers.{i}.mixer.v_proj", m.v_proj.weight)
        serialize_fp32(out_file, m.a_proj.weight)
        serialize_fp32(out_file, m.b_proj.weight)
        write_q8(f"layers.{i}.mixer.g_proj", m.g_proj.weight)
        serialize_fp32(out_file, m.q_conv1d.weight)
        serialize_fp32(out_file, m.k_conv1d.weight)
        serialize_fp32(out_file, m.v_conv1d.weight)
        serialize_fp32(out_file, -torch.exp(m.A_log.float()))
        serialize_fp32(out_file, m.dt_bias)
        serialize_fp32(out_file, m.o_norm.weight)
        write_q8(f"layers.{i}.mixer.o_proj", m.o_proj.weight)

        serialize_fp32(out_file, layer.ffn_norm.weight)
        write_q8(f"layers.{i}.ffn.gate_proj", f.gate_proj.weight)
        write_q8(f"layers.{i}.ffn.down_proj", f.down_proj.weight)
        write_q8(f"layers.{i}.ffn.up_proj", f.up_proj.weight)

    serialize_fp32(out_file, model.norm.weight)
    if not shared_classifier:
        write_q8("output", model.output.weight)

    ew.sort(reverse=True)
    print(f"max quantization group error across all weights: {ew[0][0]}")

    out_file.close()
    print(f"wrote {filepath}")

# -----------------------------------------------------------------------------
# API entrypoint

def model_export(model, filepath, version=1, group_size=32):
    """
    v1: float32 export for run.c
    v2: int8 quantized Q8_0 export for runq.c
    """
    if version == 1:
        version1_export(model, filepath)
    elif version == 2:
        version2_export(model, filepath, group_size=group_size)
    else:
        raise ValueError(f"unknown version {version}")

# -----------------------------------------------------------------------------
# checkpoint loading

def load_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu")
    model_args = checkpoint.get("model_args")
    if model_args is None:
        raise ValueError(f"{path} does not contain model_args")
    model = GDNLM(ModelArgs(**model_args))
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for key in list(state_dict.keys()):
        if key.startswith(unwanted_prefix):
            state_dict[key[len(unwanted_prefix):]] = state_dict.pop(key)
    model.load_state_dict(state_dict)
    model.eval()
    return model

# -----------------------------------------------------------------------------
# CLI entrypoint

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default=os.path.join("out", "ckpt.pt"),
                        help="model checkpoint, .pt file")
    parser.add_argument(
        "--output",
        default=os.path.join("out", "model.bin"),
                        help="the output filepath")
    parser.add_argument("--version", type=int, default=1, choices=(1, 2),
                        help="v1 fp32 for run.c; v2 Q8_0 for runq.c")
    parser.add_argument("--group-size", type=int, default=32,
                        help="group size used for int8 quantization (default: 32)")
    args = parser.parse_args()

    model = load_checkpoint(args.checkpoint)
    model_export(model, args.output, args.version, args.group_size)
