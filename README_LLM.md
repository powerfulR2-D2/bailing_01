# 百聆 LLM 配置指南

百聆项目现在支持两种 LLM (大语言模型) 配置模式：本地 Ollama 模型和 OpenAI 兼容 API。默认配置为使用本地 Ollama 模型。

## 目录

- [配置文件](#配置文件)
- [使用本地 Ollama 模型](#使用本地-ollama-模型) (默认)
- [切换到 OpenAI 兼容 API](#切换到-openai-兼容-api)
- [自定义模型配置](#自定义模型配置)
- [任务特定模型](#任务特定模型)
- [环境变量优先级](#环境变量优先级)
- [故障排除](#故障排除)

## 配置文件

百聆项目的 LLM 配置主要由两个文件控制：

1. **config.yaml** - 主要配置文件，位于 `bailing/interviewer/` 目录下
2. **.env** - 环境变量文件，用于存储敏感信息（如API密钥）和覆盖默认配置

## 使用本地 Ollama 模型

系统默认使用本地 Ollama 模型进行对话生成。这种方式不需要外部 API 密钥，所有处理都在本地完成。

### 确认本地配置

确保 `config.yaml` 文件中的 `provider` 设置为 `ollama`：

```yaml
llm:
  provider: ollama
```

### Ollama 服务设置

1. 确保在您的设备上安装并运行了 [Ollama](https://ollama.ai/)
2. 本地 Ollama 服务地址默认为：`https://a001-ollama.cpolar.cn`
3. 如需使用不同的地址，可在 `config.yaml` 或 `.env` 文件中修改

### 推荐模型

推荐使用的模型包括：
- `gemma3:27b` - 默认模型，综合性能良好
- `llama3` - 较小模型，适合资源受限设备
- `mistral` - 高性能模型，适合复杂任务

## 切换到 OpenAI 兼容 API

如需使用 OpenAI 兼容的 API 服务 (如官方 OpenAI API 或兼容服务)，请按以下步骤操作：

### 1. 修改配置文件

打开 `bailing/interviewer/config.yaml` 文件，将 `provider` 的值改为 `openai`：

```yaml
llm:
  provider: openai
```

### 2. 配置 API 凭据

在 `.env` 文件中设置以下环境变量：

```
OPENAI_API_KEY="您的API密钥"
OPENAI_BASE_URL="https://api.openai.com/v1"  # 官方API地址或兼容服务地址
```

如果您使用的是非官方兼容服务 (如 API2d 等)，请将 `OPENAI_BASE_URL` 修改为相应的服务地址。

### 3. 选择合适的模型

在 `config.yaml` 或通过环境变量设置所需的模型：

```yaml
openai:
  models:
    default: gpt-4-turbo
    decision: gpt-4-turbo
    natural_question: gpt-3.5-turbo
    reflection: gpt-4-turbo
```

## 自定义模型配置

您可以为不同类型的任务配置不同的模型，以优化性能和成本：

1. **决策生成** - 用于复杂逻辑判断，推荐使用高性能模型
2. **自然问题生成** - 生成更自然的问题表述，可使用轻量模型
3. **最终反思** - 进行深度分析和总结，推荐使用高性能模型

### 示例配置

```yaml
# 本地 Ollama 模型配置示例
ollama:
  models:
    default: gemma3:27b
    decision: mixtral:7b
    natural_question: llama3:8b
    reflection: gemma3:27b

# OpenAI 兼容 API 配置示例
openai:
  models:
    default: gpt-4-turbo
    decision: gpt-4
    natural_question: gpt-3.5-turbo
    reflection: gpt-4-turbo
```

## 环境变量优先级

环境变量优先级高于配置文件，这允许您快速切换配置而无需修改原始文件：

- `LLM_PROVIDER` - 控制使用 OpenAI API 还是本地 Ollama
- `OPENAI_API_KEY` - OpenAI API 密钥
- `OPENAI_BASE_URL` - OpenAI API 基础 URL
- `OPENAI_DEFAULT_MODEL` - 默认模型
- `LOCAL_LLM_BASE_URL` - 本地 Ollama 服务地址
- `LOCAL_LLM_DEFAULT_MODEL` - 默认本地模型

## 故障排除

### Ollama 连接问题

如果遇到 Ollama 连接问题：

1. 确认 Ollama 服务正在运行
2. 验证 `base_url` 配置是否正确
3. 检查防火墙设置是否允许连接
4. 确认所选模型已在 Ollama 中下载安装

### API 连接问题

如果遇到 API 连接问题：

1. 确认 API 密钥正确且未过期
2. 检查网络连接和代理设置
3. 验证 API 服务地址是否正确
4. 确认使用的模型名称正确且可用

如需更多帮助，请查看日志文件或提交 GitHub Issue。
