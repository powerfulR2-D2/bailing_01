from typing import List, Dict
from openai import AsyncClient
import logging
import json

class ReflectionModule:
    def __init__(self, openai_client: AsyncClient, scale_type: str = "hamd"):
        """Initialize the reflection module."""
        self.client = openai_client
        self.scale_type = scale_type

    # HAMD量表反思模板
    HAMD_REFLECTION_TEMPLATE = """
基于以下对话历史生成汉密尔顿抑郁量表(HAMD)评估报告：
{history}

输出要求：
1. 结构化摘要（JSON格式）：
   - 关键抑郁症状（最多5个）
   - 时间范围矛盾
   - 待澄清细节
2. 原始对话片段（最近6条）
3. 流程改进建议（自然语言）
"""

    # HAMA量表反思模板
    HAMA_REFLECTION_TEMPLATE = """
基于以下对话历史生成汉密尔顿焦虑量表(HAMA)评估报告：
{history}

输出要求：
1. 结构化摘要（JSON格式）：
   - 关键焦虑症状（最多5个）
   - 时间范围矛盾
   - 待澄清细节
2. 原始对话片段（最近6条）
3. 流程改进建议（自然语言）
"""

    # MINI量表反思模板
    MINI_REFLECTION_TEMPLATE = """
基于以下对话历史生成MINI简明国际神经精神访谈评估报告：
{history}

输出要求：
1. 结构化摘要（JSON格式）：
   - 关键精神症状（最多5个）
   - 时间范围矛盾
   - 待澄清细节
2. 原始对话片段（最近6条）
3. 流程改进建议（自然语言）
"""

    async def generate_reflection(self, conversation_history: List[Dict]) -> dict:
        """单次LLM调用生成完整反思报告"""
        try:
            # 保留最近8条对话
            recent_history = conversation_history[-8:]
            dialog_snippets = [f"{msg['role']}: {msg['content']}" for msg in recent_history]
            
            # 根据量表类型选择不同的模板
            if self.scale_type == "hama":
                template = self.HAMA_REFLECTION_TEMPLATE
            elif self.scale_type == "mini":
                template = self.MINI_REFLECTION_TEMPLATE
            else:  # 默认使用HAMD模板
                template = self.HAMD_REFLECTION_TEMPLATE
                
            prompt = template.format(
                history='\n'.join(dialog_snippets)
            )

            completion = await self.client.chat.completions.create(
                model="deepseek-ai/DeepSeek-V3",
                messages=[{"role": "system", "content": prompt}],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            if not completion.choices:
                return {"analysis": {}, "raw_dialog": [], "suggestions": ""}
            
            summary = json.loads(completion.choices[0].message.content)
            return {
                "analysis": dict(summary.get("structured", {})),
                "raw_dialog": list(dialog_snippets[-6:]),
                "suggestions": str(summary.get("suggestions", "")),
                "scale_type": self.scale_type
            }

        except json.JSONDecodeError:
            logging.warning("反思报告JSON解析失败")
            return {"analysis": {}, "raw_dialog": [], "suggestions": ""}
        except Exception as e:
            logging.error(f"反思生成失败: {str(e)}")
            return {"analysis": {}, "raw_dialog": [], "suggestions": ""}