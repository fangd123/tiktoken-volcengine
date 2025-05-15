import os
import re
from typing import Dict, Callable, Any, List, Optional, Tuple, Union
import requests
from volcenginesdkarkruntime import Ark

# 内置编码列表，这些编码不会被火山引擎编码器处理
BUILTIN_ENCODINGS = {
    "cl100k_base", "p50k_base", "p50k_edit", "r50k_base", "gpt2", "gpt4o"
}

class VolcengineEncoding:
    """火山引擎分词编码类，作为tiktoken.Encoding的替代实现"""
    
    def __init__(self, model: str, api_endpoint: str = None):
        # 存储原始模型名称（可能带有volcengine-前缀）
        self.original_model = model
        
        # 实际发送给API的模型名称，去除前缀
        self.api_model_name = model
        if model.startswith("volcengine-"):
            self.api_model_name = model[len("volcengine-"):]
            
        # tiktoken用于识别的编码器名称
        self.name = f"volcengine-{self.api_model_name}" if not model.startswith("volcengine-") else model
        
        # API密钥和客户端
        self.api_key = os.environ.get("ARK_API_KEY")
        self.client = Ark(api_key=self.api_key)
        
        # API端点
        self.api_endpoint = api_endpoint or "https://ark.cn-beijing.volces.com/api/v3/tokenization"

    def encode(self, text: str, allowed_special: Union[set, str] = "all", disallowed_special: Union[set, str] = "") -> List[int]:
        """将文本编码为token ids
        
        参数:
            text: 要编码的文本
            allowed_special: 与tiktoken兼容的参数，但在此实现中未使用
            disallowed_special: 与tiktoken兼容的参数，但在此实现中未使用
        返回:
            token_ids列表
        """
        resp = self.client.tokenization.create(
            model=self.api_model_name,  # 使用不带前缀的模型名称
            text=[text],
        )
        if resp and "data" in resp and len(resp["data"]) > 0:
            return resp["data"][0]["token_ids"]
        return []
    
    def encode_batch(self, texts: List[str], allowed_special: Union[set, str] = "all", disallowed_special: Union[set, str] = "") -> List[List[int]]:
        """批量编码多个文本
        
        参数:
            texts: 要编码的文本列表
            allowed_special: 与tiktoken兼容的参数，但在此实现中未使用
            disallowed_special: 与tiktoken兼容的参数，但在此实现中未使用
        返回:
            每个文本的token_ids列表的列表
        """
        resp = self.client.tokenization.create(
            model=self.api_model_name,  # 使用不带前缀的模型名称
            text=texts,
        )
        if resp and "data" in resp:
            return [item["token_ids"] for item in resp["data"]]
        return [[] for _ in texts]
    
    def decode(self, tokens: List[int]) -> str:
        """将token ids解码为文本
        
        注意：火山引擎API没有直接提供解码功能，这里需要实现自定义逻辑
        或者联系火山引擎获取解码接口
        
        参数:
            tokens: 要解码的token_ids列表
        返回:
            解码后的文本
        """
        return "[火山引擎暂不支持直接解码]"
    
    def count_tokens(self, text: str) -> int:
        """计算文本包含的token数量
        
        参数:
            text: 要计算的文本
        返回:
            token数量
        """
        tokens = self.encode(text)
        return len(tokens)

def create_volcengine_encoding(model_name: str, api_endpoint: str = None) -> Dict[str, Any]:
    """创建火山引擎编码配置
    
    参数:
        model_name: 模型名称 (可能带有volcengine-前缀)
        api_endpoint: API端点URL
    返回:
        编码配置字典
    """
    return {
        "name": model_name,
        "pat_str": None,  # 使用火山引擎API，不需要pat_str
        "mergeable_ranks": None,  # 使用火山引擎API，不需要mergeable_ranks
        "special_tokens": {},  # 使用火山引擎API，不需要special_tokens
        "explicit_n_vocab": 100000,  # 假设的词汇表大小
        "encoder_class": VolcengineEncoding,
        "encoder_kwargs": {
            "model": model_name,
            "api_endpoint": api_endpoint
        }
    }

def get_encoding_for_model(model_name: str) -> Dict[str, Any]:
    """为任何模型创建编码器
    
    参数:
        model_name: 模型名称
    返回:
        编码配置字典
    """
    # 如果是内置编码，返回None，让tiktoken使用其内置处理
    if model_name in BUILTIN_ENCODINGS:
        return None
    
    # 对于以volcengine-开头的模型，直接使用
    if model_name.startswith("volcengine-"):
        return create_volcengine_encoding(model_name)
    
    # 对于其他模型，添加volcengine-前缀
    return create_volcengine_encoding(f"volcengine-{model_name}")

# 注册编码器构造函数 - 预定义的模型
ENCODING_CONSTRUCTORS: Dict[str, Callable[[], Dict[str, Any]]] = {
    "volcengine-doubao-pro-32k-241215": lambda: create_volcengine_encoding("volcengine-doubao-pro-32k-241215"),
}

# 添加动态模型处理函数
def __getattr__(name: str) -> Any:
    """处理任何未知的编码名称请求
    
    参数:
        name: 被请求的属性名称
    返回:
        如果是encoding_for_model请求，返回动态创建的编码
    """
    if name == "encoding_for_model":
        return get_encoding_for_model
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'") 