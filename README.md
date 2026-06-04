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

数据统一下载到仓库根目录的 `data/`：

```bash
python tinystories.py download
```

### 2) 预分词

使用仓库内置的 `tok512`（vocab_size=512）：

```bash
python tinystories.py pretokenize --vocab_size=512
```

使用 Llama 2 原始词表（vocab_size=32000，适合 15M 及以上规模）：

```bash
python tinystories.py pretokenize
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


