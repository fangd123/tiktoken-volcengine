import os
import tiktoken
from volcenginesdkarkruntime import Ark

# 设置API密钥（实际使用中请替换为你的密钥）
os.environ["ARK_API_KEY"] = "your_api_key_here"

def print_separator(title):
    """打印分隔线"""
    print("\n" + "="*50)
    print(f" {title} ".center(50, "="))
    print("="*50 + "\n")

# 1. 直接使用火山引擎API
print_separator("直接使用火山引擎API")
client = Ark(api_key=os.environ.get("ARK_API_KEY"))
resp = client.tokenization.create(
    model="doubao-pro-32k-241215",  # 注意：直接使用API时不需要前缀
    text=["天空为什么这么蓝", "花儿为什么这么香"],
)
print(resp)

# 2. 使用扩展的tiktoken（安装后）
print_separator("使用扩展的tiktoken")
try:
    # 2.1 获取火山引擎编码器 - 使用预定义的模型名称
    print("【方式1: 使用预定义的模型名称】")
    enc = tiktoken.get_encoding("volcengine-doubao-pro-32k-241215")
    # 内部会自动去除前缀，实际API调用时使用 "doubao-pro-32k-241215"
    
    text = "天空为什么这么蓝"
    tokens = enc.encode(text)
    print(f"编码结果: {tokens}")
    print(f"实际使用的API模型名称: {enc.api_model_name}")  # 这里显示的是不带前缀的名称
    print(f"tiktoken编码器名称: {enc.name}\n")  # 这里显示的是带前缀的名称
    
    # 2.2 使用encoding_for_model函数 - 直接使用模型名称
    print("【方式2: 直接使用模型名称】")
    enc2 = tiktoken.encoding_for_model("doubao-pro-32k-241215")  # 自动添加volcengine-前缀
    # tiktoken内部会将其转换为 "volcengine-doubao-pro-32k-241215"，但API调用时使用 "doubao-pro-32k-241215"
    
    tokens2 = enc2.encode(text)
    print(f"编码结果: {tokens2}")
    print(f"实际使用的API模型名称: {enc2.api_model_name}")  # 这里显示的是不带前缀的名称
    print(f"tiktoken编码器名称: {enc2.name}\n")  # 这里显示的是带前缀的名称
    
    # 2.3 使用任意模型名称
    print("【方式3: 使用任意模型名称】")
    # 注意：这会尝试使用指定的模型名称调用火山引擎API，如果模型不存在会抛出错误
    custom_model = "some-other-model"
    enc3 = tiktoken.encoding_for_model(custom_model)  # 自动添加volcengine-前缀
    
    try:
        tokens3 = enc3.encode(text)
        print(f"编码结果: {tokens3}")
        print(f"实际使用的API模型名称: {enc3.api_model_name}")  # 这里显示的是不带前缀的名称
        print(f"tiktoken编码器名称: {enc3.name}\n")  # 这里显示的是带前缀的名称
    except Exception as e:
        print(f"模型不存在或其他错误: {e}")
        print(f"尝试使用的API模型名称: {enc3.api_model_name}")
        print(f"尝试访问的API端点: {enc3.api_endpoint}\n")
    
    # 2.4 尝试批量编码
    print("【批量编码示例】")
    texts = ["天空为什么这么蓝", "花儿为什么这么香"]
    batch_tokens = enc.encode_batch(texts)
    print(f"批量编码结果: {batch_tokens}\n")
    
except Exception as e:
    print(f"使用tiktoken扩展时出错: {e}")
    print("请确保已安装扩展包: pip install ./volcengine_tiktoken_extension")
    
print_separator("使用内置模型（不受影响）")
try:
    # 3.1 使用tiktoken内置编码
    enc_builtin = tiktoken.get_encoding("cl100k_base")
    text = "Hello, world!"
    tokens = enc_builtin.encode(text)
    decoded = enc_builtin.decode(tokens)
    print(f"内置编码器: cl100k_base")
    print(f"编码 '{text}' 结果: {tokens}")
    print(f"解码结果: '{decoded}'")
except Exception as e:
    print(f"使用内置编码器时出错: {e}")
    
print_separator("使用说明")
print("如果你看到错误，请确保:")
print("1. 已安装tiktoken: pip install tiktoken")
print("2. 已安装火山引擎SDK: pip install volcenginesdkarkruntime")
print("3. 已安装此扩展: pip install ./volcengine_tiktoken_extension")
print("4. 设置了有效的ARK_API_KEY环境变量")
print("\n提示: 火山引擎API端点使用的是不带前缀的模型名称，如 'doubao-pro-32k-241215'")
print("而不是带有前缀的tiktoken编码器名称，如 'volcengine-doubao-pro-32k-241215'") 