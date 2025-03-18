from typing import List, Dict
from openai import AsyncClient
import logging
import json
import re

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
        # 创建默认结构，始终返回此结构避免解析错误
        default_analysis = {
            "summary": {
                "key_symptoms": [],
                "time_info": [],
                "unclear_details": []
            },
            "key_points": {
                "covered": [],
                "pending": []
            }
        }
        
        default_result = {
            "analysis": default_analysis,
            "raw_dialog": [],
            "suggestions": "",
            "scale_type": self.scale_type
        }
        
        # 如果对话历史为空，直接返回默认值
        if not conversation_history:
            return default_result
        
        try:
            # 当对话历史记录太短时，不生成反思
            if len(conversation_history) < 4:
                return default_result
                
            # 最新的几条对话
            recent_history = conversation_history[-15:] if len(conversation_history) > 15 else conversation_history
            dialog_snippets = [f"{msg.get('role', 'unknown')}: {msg.get('content', '')}" for msg in recent_history if msg.get('content')]
            
            # 为避免解析问题，使用提取要点而非生成JSON的方式
            # 这是一个彻底不同的方法，可以避免JSON解析问题
            extract_template = """
分析以下对话，并回答问题：

{history}

请用简单列表回答以下问题（不要用JSON格式）：

1. 患者提到了哪些症状？
2. 患者提到了哪些时间信息？
3. 还有哪些需要澄清的信息？
4. 对话已覆盖了哪些关键点？
5. 还有哪些关键点需要进一步了解？

请直接回答，每个问题一行一个列表，格式如下：
症状：症状1，症状2
时间：时间点1，时间点2
待澄清：问题1，问题2
已覆盖：要点1，要点2
待了解：要点1，要点2
"""
            
            prompt = extract_template.format(
                history='\n'.join(dialog_snippets)
            )
            
            try:
                # 调用大语言模型
                completion = await self.client.chat.completions.create(
                    model="deepseek-ai/DeepSeek-V3",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=800
                )
                
                if not completion.choices or not completion.choices[0].message:
                    logging.warning("反思生成失败：LLM未返回选择")
                    return default_result
                
                # 获取返回的文本
                text_response = completion.choices[0].message.content.strip()
                logging.debug(f"反思原始响应: {text_response[:200]}")
                
                # 提取每个部分的列表
                symptoms = []
                times = []
                unclear = []
                covered = []
                pending = []
                
                # 使用正则表达式提取每一类信息
                # 症状
                if "症状：" in text_response:
                    symptoms_match = re.search(r"症状：(.*?)(?=\n|$)", text_response)
                    if symptoms_match:
                        symptoms_text = symptoms_match.group(1).strip()
                        symptoms = [s.strip() for s in symptoms_text.split("，") if s.strip()]
                
                # 时间
                if "时间：" in text_response:
                    times_match = re.search(r"时间：(.*?)(?=\n|$)", text_response)
                    if times_match:
                        times_text = times_match.group(1).strip()
                        times = [t.strip() for t in times_text.split("，") if t.strip()]
                
                # 待澄清
                if "待澄清：" in text_response:
                    unclear_match = re.search(r"待澄清：(.*?)(?=\n|$)", text_response)
                    if unclear_match:
                        unclear_text = unclear_match.group(1).strip()
                        unclear = [u.strip() for u in unclear_text.split("，") if u.strip()]
                
                # 已覆盖
                if "已覆盖：" in text_response:
                    covered_match = re.search(r"已覆盖：(.*?)(?=\n|$)", text_response)
                    if covered_match:
                        covered_text = covered_match.group(1).strip()
                        covered = [c.strip() for c in covered_text.split("，") if c.strip()]
                
                # 待了解
                if "待了解：" in text_response:
                    pending_match = re.search(r"待了解：(.*?)(?=\n|$)", text_response)
                    if pending_match:
                        pending_text = pending_match.group(1).strip()
                        pending = [p.strip() for p in pending_text.split("，") if p.strip()]
                
                # 创建分析结果
                analysis = {
                    "summary": {
                        "key_symptoms": symptoms,
                        "time_info": times,
                        "unclear_details": unclear
                    },
                    "key_points": {
                        "covered": covered,
                        "pending": pending
                    }
                }
                
                # 生成默认建议
                suggestions = "继续询问患者的症状持续时间和严重程度。"
                
                # 记录提取的内容
                logging.info(f"反思分析 - 症状: {symptoms}")
                logging.info(f"反思分析 - 时间: {times}")
                logging.info(f"反思分析 - 待澄清: {unclear}")
                logging.info(f"反思分析 - 已覆盖要点: {covered}")
                logging.info(f"反思分析 - 待了解要点: {pending}")
                
                return {
                    "analysis": analysis,
                    "raw_dialog": dialog_snippets[-6:] if dialog_snippets else [],
                    "suggestions": suggestions,
                    "scale_type": self.scale_type
                }
            
            except Exception as api_error:
                logging.error(f"反思生成API调用失败: {str(api_error)}")
                return default_result
        
        except Exception as e:
            logging.error(f"反思生成失败: {str(e)}")
            return default_result