# gdn.c

GDN（Gated Delta Net）架构实现：线性注意力 + 门控状态更新，训练后导出为纯 C 推理权重。

所有命令默认从本子目录执行：

```bash
cd /home/ruhai/Projects/gpts.c/gdn.c
```

创建虚拟环境
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 训练

### 1) 准备数据

数据统一下载到仓库根目录的 `data/`。当前默认数据集为 **ClimbMix**（parquet 分片）：

```bash
# 下载 170 个训练分片 + 1 个 eval 分片（最后一个分片固定作验证集）
python climbmix.py download -n 170
```

也可下载 TinyStories（旧数据集）：

```bash
python tinystories.py download
```

> **切换数据集**：`train.py` 顶部的 `from climbmix import Task` 决定使用哪个数据集。
> 想换数据集就手动改这一行（例如改成 `from tinystories import Task`）。两个数据集的
> `Task` 接口一致，除此之外无需改动 `train.py`。

### 2) 预分词

使用 Llama 2 原始词表（vocab_size=32000，适合 15M 及以上规模）：

```bash
python tinystories.py pretokenize
```
```bash
python climbmix.py pretokenize
```

使用自训练词表（vocab_size=512，例如配合 260K 模型，主要用于 TinyStories）：

```bash
python tinystories.py pretokenize --vocab_size=512
```

### 3) 训练

训练输出写入 `outputs/<size>/`，训练中自动保存 `ckpt.pt` 和 `model.bin`。

#### 260K（~310K 参数，vocab=512）

```bash
python train.py \
  --out_dir='outputs/260K' \
  --max_iters=100000 \
  --batch_size=128 \
  --gradient_accumulation_steps=1 \
  --device='cuda' \
  --dtype='float32' \
  --compile=True \
  --max_seq_len=512 \
  --vocab_source='custom' \
  --vocab_size=512 \
  --learning_rate=1e-3 \
  --warmup_iters=1000 \
  --beta2=0.99 \
  --dim=64 \
  --n_layers=5 \
  --num_heads=8 \
  --head_k_dim=8 \
  --head_v_dim=8 \
  --hidden_dim=172 \
  --conv_size=4
```

#### 15M（~15M 参数，vocab=32000）

```bash
python train.py
# 多卡环境
torchrun --standalone --nproc_per_node=8 train.py --out_dir=outputs/climbmix15M_50B_100k 
```

#### 42M（~42M 参数，vocab=32000）

```bash
torchrun --standalone --nproc_per_node=8 train.py \
  --out_dir=outputs/climbmix42M_50B_100k \
  --dim=512 --n_layers=8 --num_heads=8 \
  --head_k_dim=64 --head_v_dim=64 --hidden_dim=1376 --conv_size=4 \
  --max_seq_len=1024 \
  --batch_size=64 --gradient_accumulation_steps=8 \
  --learning_rate=5e-4 --max_iters=100000 --warmup_iters=1000 \
  --weight_decay=0.1 --beta1=0.9 --beta2=0.95 --grad_clip=1.0 \
  --dtype=bfloat16 --compile=False \
  --eval_interval=2000 --eval_iters=100
```

#### 110M（~110M 参数，vocab=32000）

```bash
torchrun --standalone --nproc_per_node=8 train.py \
  --out_dir=outputs/climbmix110M_50B_100k \
  --dim=768 --n_layers=12 --num_heads=12 \
  --head_k_dim=64 --head_v_dim=64 --hidden_dim=2048 --conv_size=4 \
  --max_seq_len=1024 \
  --batch_size=64 --gradient_accumulation_steps=8 \
  --learning_rate=5e-4 --max_iters=100000 --warmup_iters=1000 \
  --weight_decay=0.1 --beta1=0.9 --beta2=0.95 --grad_clip=1.0 \
  --dtype=bfloat16 --compile=False \
  --eval_interval=2000 --eval_iters=100
```

---

## C 推理

### 编译

```bash
make
```

### 运行

```bash
# 260K 模型
./run outputs/260K/model.bin -z ../tok512.bin -t 0 -p 0.9 -s 1337 -n 128 -i "Once upon a time,"

# 15M 模型（使用 Llama 2 tokenizer）
./run outputs/15M/model.bin -z ../tokenizer.bin -t 0 -p 0.9 -s 1337 -n 128 -i "Once upon a time,"
```

`-z` 指定 tokenizer，`-t` 温度，`-p` top-p，`-s` 随机种子，`-n` 生成 token 数，`-i` 提示词。

## CORE 评测

[CORE](https://arxiv.org/abs/2406.11717) benchmark（DCLM 论文指标），用于评估 base 模型的综合能力。评测脚本为 `core_eval.py`，首次运行会自动下载 `data/eval_bundle/`。

```bash
# 指定训练产出的 checkpoint（.pt）
python core_eval.py --checkpoint outputs/climbmix15M_50B_100k/climbmix15M.pt
python core_eval.py --checkpoint outputs/climbmix42M_50B_100k/climbmix42M.pt
python core_eval.py --checkpoint outputs/climbmix110M_50B_100k/climbmix110M.pt
```

结果写入 checkpoint 同目录：

- `<checkpoint_stem>.bin`：导出的 C 推理权重
- `core_eval.csv`：各子任务 Accuracy / Centered 分数及 CORE 汇总

可选参数：`--tokenizer`（默认 `tokenizer.model`）、`--device`（默认 `cuda`）、`--max_per_task`（每任务最多评测样本数，`-1` 为全部）。

### 评测结果

ClimbMix 上训练 100k iter（~50B tokens）后的 CORE benchmark 结果：

| Task | 15M Acc | 42M Acc | 110M Acc | 15M Centered | 42M Centered | 110M Centered |
|------|---------|---------|----------|--------------|--------------|---------------|
| hellaswag_zeroshot | 0.2815 | 0.3229 | 0.3971 | 0.0420 | 0.0973 | 0.1962 |
| jeopardy | 0.0014 | 0.0043 | 0.0132 | 0.0014 | 0.0043 | 0.0132 |
| bigbench_qa_wikidata | 0.0842 | 0.2008 | 0.3148 | 0.0842 | 0.2008 | 0.3148 |
| arc_easy | 0.3851 | 0.4533 | 0.5434 | 0.1801 | 0.2710 | 0.3911 |
| arc_challenge | 0.2270 | 0.2449 | 0.2756 | -0.0307 | -0.0068 | 0.0341 |
| copa | 0.4200 | 0.4900 | 0.5100 | -0.1600 | -0.0200 | 0.0200 |
| commonsense_qa | 0.2097 | 0.3170 | 0.2056 | 0.0121 | 0.1462 | 0.0070 |
| piqa | 0.6007 | 0.6273 | 0.6882 | 0.2013 | 0.2546 | 0.3765 |
| openbook_qa | 0.2780 | 0.2940 | 0.3300 | 0.0373 | 0.0587 | 0.1067 |
| lambada_openai | 0.1304 | 0.2141 | 0.3060 | 0.1304 | 0.2141 | 0.3060 |
| hellaswag | 0.2754 | 0.3189 | 0.3917 | 0.0339 | 0.0918 | 0.1889 |
| winograd | 0.5018 | 0.5641 | 0.5604 | 0.0037 | 0.1282 | 0.1209 |
| winogrande | 0.4909 | 0.4988 | 0.5209 | -0.0182 | -0.0024 | 0.0418 |
| bigbench_dyck_languages | 0.0110 | 0.0320 | 0.1500 | 0.0110 | 0.0320 | 0.1500 |
| agi_eval_lsat_ar | 0.2565 | 0.2217 | 0.2696 | 0.0707 | 0.0272 | 0.0870 |
| bigbench_cs_algorithms | 0.3947 | 0.4197 | 0.4508 | 0.3947 | 0.4197 | 0.4508 |
| bigbench_operators | 0.1381 | 0.1286 | 0.1476 | 0.1381 | 0.1286 | 0.1476 |
| bigbench_repeat_copy_logic | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| squad | 0.0099 | 0.0346 | 0.0645 | 0.0099 | 0.0346 | 0.0645 |
| coqa | 0.0441 | 0.0864 | 0.1156 | 0.0441 | 0.0864 | 0.1156 |
| boolq | 0.5196 | 0.4976 | 0.5450 | -0.2643 | -0.3222 | -0.1975 |
| bigbench_language_identification | 0.2543 | 0.2514 | 0.2547 | 0.1796 | 0.1765 | 0.1801 |
| **CORE** | — | — | — | **0.0501** | **0.0918** | **0.1416** |

