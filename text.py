import requests
import json

# --- 配置 ---
OLLAMA_API_URL = "https://a001-ollama.cpolar.cn/api/generate"
MODEL_NAME = "gemma3:27b"
PROMPT = "给我完整的滕王阁序"

# --- 准备请求数据 ---
payload = {
    "model": MODEL_NAME,
    "prompt": PROMPT,
    "stream": True,  # 设置为 True 获取流式响应
    "options": {
        "temperature": 0.8,        # 尝试增加一点随机性
        "repetition_penalty": 1.2, # 增加重复惩罚 (关键!)
        # "top_k": 40,             # 可以尝试调整或注释掉
        # "top_p": 0.9             # 可以尝试调整或注释掉
    }
}

# --- 发送 POST 请求并处理流式响应 ---
try:
    print(f"正在向 {OLLAMA_API_URL} 发送流式请求...")
    # 使用 stream=True 参数
    with requests.post(
        OLLAMA_API_URL,
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload),
        stream=True,
        timeout=180
    ) as response:

        response.raise_for_status() # 检查初始连接状态

        print("\n--- 模型响应 (流式) ---")
        full_response = ""
        # 迭代处理响应内容行
        for line in response.iter_lines():
            if line:
                try:
                    # 解码每一行 (Ollama 流式响应每行是一个 JSON 对象)
                    chunk = json.loads(line.decode('utf-8'))
                    # 提取 'response' 部分的文本
                    response_part = chunk.get('response', '')
                    print(response_part, end='', flush=True) # 实时打印，不换行
                    full_response += response_part

                    # 检查生成是否完成 (流式响应的最后一条会包含 'done': True)
                    if chunk.get('done'):
                        print("\n--- 流式响应结束 ---")
                        # 可以选择在这里打印整个完整响应
                        # print("\n完整响应:\n", full_response)
                        # 也可以打印最后一条包含的统计信息等
                        # print("\n统计信息:", chunk)
                        break
                except json.JSONDecodeError:
                    print(f"\n警告：无法解析流中的某一行: {line.decode('utf-8')}")
                except Exception as e_chunk:
                    print(f"\n处理流式块时出错: {e_chunk}")

except requests.exceptions.Timeout as e:
    print(f"请求超时错误: {e}")
except requests.exceptions.RequestException as e:
    print(f"请求 Ollama API 时出错: {e}")
except Exception as e:
    print(f"发生意外错误: {e}")