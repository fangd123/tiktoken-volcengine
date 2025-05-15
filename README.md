# ⏳ tiktoken

tiktoken is a fast [BPE](https://en.wikipedia.org/wiki/Byte_pair_encoding) tokeniser for use with
OpenAI's models.

```python
import tiktoken
enc = tiktoken.get_encoding("o200k_base")
assert enc.decode(enc.encode("hello world")) == "hello world"

# To get the tokeniser corresponding to a specific model in the OpenAI API:
enc = tiktoken.encoding_for_model("gpt-4o")
```

The open source version of `tiktoken` can be installed from [PyPI](https://pypi.org/project/tiktoken):
```
pip install tiktoken
```

The tokeniser API is documented in `tiktoken/core.py`.

Example code using `tiktoken` can be found in the
[OpenAI Cookbook](https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb).


## Performance

`tiktoken` is between 3-6x faster than a comparable open source tokeniser:

![image](https://raw.githubusercontent.com/openai/tiktoken/main/perf.svg)

Performance measured on 1GB of text using the GPT-2 tokeniser, using `GPT2TokenizerFast` from
`tokenizers==0.13.2`, `transformers==4.24.0` and `tiktoken==0.2.0`.


## Getting help

Please post questions in the [issue tracker](https://github.com/openai/tiktoken/issues).

If you work at OpenAI, make sure to check the internal documentation or feel free to contact
@shantanu.

## What is BPE anyway?

Language models don't see text like you and I, instead they see a sequence of numbers (known as tokens).
Byte pair encoding (BPE) is a way of converting text into tokens. It has a couple desirable
properties:
1) It's reversible and lossless, so you can convert tokens back into the original text
2) It works on arbitrary text, even text that is not in the tokeniser's training data
3) It compresses the text: the token sequence is shorter than the bytes corresponding to the
   original text. On average, in practice, each token corresponds to about 4 bytes.
4) It attempts to let the model see common subwords. For instance, "ing" is a common subword in
   English, so BPE encodings will often split "encoding" into tokens like "encod" and "ing"
   (instead of e.g. "enc" and "oding"). Because the model will then see the "ing" token again and
   again in different contexts, it helps models generalise and better understand grammar.

`tiktoken` contains an educational submodule that is friendlier if you want to learn more about
the details of BPE, including code that helps visualise the BPE procedure:
```python
from tiktoken._educational import *

# Train a BPE tokeniser on a small amount of text
enc = train_simple_encoding()

# Visualise how the GPT-4 encoder encodes text
enc = SimpleBytePairEncoding.from_tiktoken("cl100k_base")
enc.encode("hello world aaaaaaaaaaaa")
```


## Extending tiktoken

You may wish to extend `tiktoken` to support new encodings. There are two ways to do this.


**Create your `Encoding` object exactly the way you want and simply pass it around.**

```python
cl100k_base = tiktoken.get_encoding("cl100k_base")

# In production, load the arguments directly instead of accessing private attributes
# See openai_public.py for examples of arguments for specific encodings
enc = tiktoken.Encoding(
    # If you're changing the set of special tokens, make sure to use a different name
    # It should be clear from the name what behaviour to expect.
    name="cl100k_im",
    pat_str=cl100k_base._pat_str,
    mergeable_ranks=cl100k_base._mergeable_ranks,
    special_tokens={
        **cl100k_base._special_tokens,
        "<|im_start|>": 100264,
        "<|im_end|>": 100265,
    }
)
```

**Use the `tiktoken_ext` plugin mechanism to register your `Encoding` objects with `tiktoken`.**

This is only useful if you need `tiktoken.get_encoding` to find your encoding, otherwise prefer
option 1.

To do this, you'll need to create a namespace package under `tiktoken_ext`.

Layout your project like this, making sure to omit the `tiktoken_ext/__init__.py` file:
```
my_tiktoken_extension
├── tiktoken_ext
│   └── my_encodings.py
└── setup.py
```

`my_encodings.py` should be a module that contains a variable named `ENCODING_CONSTRUCTORS`.
This is a dictionary from an encoding name to a function that takes no arguments and returns
arguments that can be passed to `tiktoken.Encoding` to construct that encoding. For an example, see
`tiktoken_ext/openai_public.py`. For precise details, see `tiktoken/registry.py`.

Your `setup.py` should look something like this:
```python
from setuptools import setup, find_namespace_packages

setup(
    name="my_tiktoken_extension",
    packages=find_namespace_packages(include=['tiktoken_ext*']),
    install_requires=["tiktoken"],
    ...
)
```

Then simply `pip install ./my_tiktoken_extension` and you should be able to use your
custom encodings! Make sure **not** to use an editable install.

## 火山引擎大模型分词扩展

这是一个为火山引擎方舟大模型平台(ARK)提供的tiktoken扩展。它允许你使用tiktoken API访问火山引擎的分词功能。

### 安装

1. 克隆此仓库
2. 安装扩展:
   ```bash
   pip install ./volcengine_tiktoken_extension
   ```

### 使用方法

```python
import os
import tiktoken

# 设置API密钥
os.environ["ARK_API_KEY"] = "your_api_key_here"

# 方式1: 使用预定义的编码名称
enc = tiktoken.get_encoding("volcengine-doubao-pro-32k-241215")
# 注意: 虽然编码器名称带有前缀，但实际API调用使用的是不带前缀的模型名称

# 方式2: 直接使用模型名称（自动添加前缀）
enc = tiktoken.encoding_for_model("doubao-pro-32k-241215")
# 注意: 虽然传入的是不带前缀的名称，但会自动添加前缀成为"volcengine-doubao-pro-32k-241215"
# 实际API调用时仍使用不带前缀的名称"doubao-pro-32k-241215"

# 方式3: 使用任意模型名称 
# 任意有效的火山引擎模型都可以直接使用
enc = tiktoken.encoding_for_model("your-custom-model-name")

# 编码文本
text = "天空为什么这么蓝"
tokens = enc.encode(text)
print(f"编码结果: {tokens}")

# 批量编码
texts = ["天空为什么这么蓝", "花儿为什么这么香"]
batch_tokens = enc.encode_batch(texts)
print(f"批量编码结果: {batch_tokens}")

# 查看实际使用的API模型名称（不带前缀）
print(f"API模型名称: {enc.api_model_name}")

# 查看tiktoken编码器名称（带前缀）
print(f"编码器名称: {enc.name}")
```

### 模型名称说明

本扩展在内部和API调用之间维护了两套命名方式:

1. **tiktoken编码器名称**：带有`volcengine-`前缀，如`volcengine-doubao-pro-32k-241215`
   - 这是tiktoken内部使用的名称，用于在tiktoken中唯一标识编码器

2. **API模型名称**：不带前缀，如`doubao-pro-32k-241215`
   - 这是实际发送给火山引擎API的模型名称
   - 扩展会自动去除前缀后再调用API

### 支持的模型

扩展支持以下两种使用模式:

1. **预定义模型**：目前内置支持以下火山引擎模型:
   - doubao-pro-32k-241215

2. **动态支持任意模型**：
   - 任何非tiktoken内置模型的请求都会自动使用火山引擎API进行处理
   - 可以直接使用模型名称，无需添加前缀
   - 内置编码（如cl100k_base, p50k_base等）不受影响

### 限制

- 目前解码功能不完整，因为火山引擎API没有提供直接的解码功能。
- 仅支持基本的编码功能，不支持完整的BPE操作。
- 使用非预定义模型时，请确保该模型在火山引擎平台上存在。
- 默认使用`https://ark.cn-beijing.volces.com/api/v3/tokenization`作为API端点。

### 更多信息

有关火山引擎方舟大模型平台的更多信息，请参阅[火山引擎官方文档](https://www.volcengine.com/docs/82379)。

