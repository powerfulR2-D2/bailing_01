import json
import os
import logging
from typing import Dict, List, Optional
from openai import AsyncClient

class QuestionGenerator:
    def __init__(self, openai_client: AsyncClient):
        self.client = openai_client
        
    async def generate_questions(self, patient_info: str) -> List[Dict]:
        """Generate personalized interview questions based on patient information."""
        system_prompt = """你是一个专业的心理健康评估专家。你需要根据患者的个人信息，生成针对性的HAMD抑郁量表评估问题。
        这些问题应该涵盖HAMD量表的核心评估领域，包括：抑郁情绪、内疚感、自杀、入睡困难、睡眠不深、早醒、工作和兴趣、
        精神运动迟滞、激越、精神性焦虑、躯体性焦虑、胃肠道症状、全身症状、性症状、疑病、体重减轻、自知力。
        生成的问题要更加个性化，要基于患者的具体情况。每个问题必须包含以下字段：
        - id: 问题的唯一标识符
        - question: 问题内容
        - type: 问题类型 (instruction/open_ended)
        - expected_topics: 预期回答涉及的话题列表
        - follow_up_questions: 可能的追问列表"""
        
        user_prompt = f"""请根据以下患者信息生成个性化的HAMD评估问题：

        患者信息：
        {patient_info}

        请确保生成的问题格式如下：
        {{
            "questions": [
                {{
                    "id": "greeting",
                    "question": "开场白...",
                    "type": "instruction"
                }},
                {{
                    "id": "depressed_mood",
                    "question": "个性化问题...",
                    "type": "open_ended",
                    "expected_topics": ["话题1", "话题2"],
                    "follow_up_questions": ["追问1", "追问2"]
                }},
                ...
            ]
        }}

        注意：
        1. 第一个问题必须是greeting类型的开场白
        2. 问题内容要基于患者信息进行个性化
        3. 确保JSON格式正确，所有字段都必须存在
        4. 每个问题都要有合适的follow_up_questions用于追问
        5. 确保足够的问题数量能完成有效的HAMD抑郁评估
        """
        
        try:
            logging.info("Generating questions using GPT-4o...")
            response = await self.client.chat.completions.create(
                model="gpt-4o", # 不要修改gpt-4o，这是正确的模型名字
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7
            )
            
            # 解析响应并验证格式
            response_text = response.choices[0].message.content
            logging.info(f"Raw response: {response_text}")
            
            try:
                # 清理响应文本，移除可能的markdown标记
                cleaned_response = response_text.strip()
                if cleaned_response.startswith('```json'):
                    cleaned_response = cleaned_response[7:]  # 移除开头的 ```json
                if cleaned_response.endswith('```'):
                    cleaned_response = cleaned_response[:-3]  # 移除结尾的 ```
                cleaned_response = cleaned_response.strip()
                
                questions = json.loads(cleaned_response)
                if not isinstance(questions, dict) or "questions" not in questions:
                    logging.error("Invalid response format")
                    raise ValueError("Invalid response format")
                
                # 验证问题格式
                for q in questions["questions"]:
                    required_fields = ["id", "question", "type"]
                    if q["type"] == "open_ended":
                        required_fields.extend(["expected_topics", "follow_up_questions"])
                    
                    for field in required_fields:
                        if field not in q:
                            raise ValueError(f"Missing required field: {field} in question {q.get('id', 'unknown')}")
                
                logging.info(f"Successfully generated {len(questions['questions'])} questions")
                
                # 保存生成的问题到临时文件
                temp_script_path = os.path.join(os.path.dirname(__file__), "temp_script.json")
                with open(temp_script_path, 'w', encoding='utf-8') as f:
                    json.dump(questions, f, ensure_ascii=False, indent=2)
                logging.info(f"Saved generated questions to {temp_script_path}")
                
                return questions["questions"]
                
            except json.JSONDecodeError as e:
                logging.error(f"Failed to parse response as JSON: {str(e)}")
                raise
            
        except Exception as e:
            logging.error(f"Error generating questions: {str(e)}")
            # 返回默认问题
            default_script_path = os.path.join(os.path.dirname(__file__), "default_script.json")
            with open(default_script_path, 'r', encoding='utf-8') as f:
                default_questions = json.load(f)
                logging.info("Using default questions due to error")
                return default_questions["questions"]
    
    @staticmethod
    def read_patient_info_file(file_path: str) -> str:
        """Read patient information from a text file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except Exception as e:
            logging.error(f"Error reading patient info file: {str(e)}")
            raise
