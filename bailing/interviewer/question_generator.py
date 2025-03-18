import json
import os
import logging
from typing import Dict, List, Optional
from openai import AsyncClient

class QuestionGenerator:
    def __init__(self, openai_client: AsyncClient):
        self.client = openai_client
        
    async def generate_questions(self, patient_info: str, scale_type: str = "hamd") -> List[Dict]:
        """Generate personalized interview questions based on patient information.
        
        Args:
            patient_info: A string containing patient information
            scale_type: The type of assessment scale ("hamd", "hama", or "mini")
            
        Returns:
            A list of question dictionaries
        """
        # 根据不同量表类型选择合适的系统提示
        if scale_type.lower() == "mini":
            system_prompt = self._get_mini_system_prompt()
        elif scale_type.lower() == "hama":
            system_prompt = self._get_hama_system_prompt()
        else:  # 默认使用HAMD
            system_prompt = self._get_hamd_system_prompt()
            
        logging.info(f"Using {scale_type.upper()} system prompt for question generation")
        
        user_prompt = f"""
        请根据以下患者信息生成个性化的初始评估问题：

患者信息：
{patient_info}

请严格遵循system_prompt的要求生成JSON格式的问题列表，并仔细检查确保所有评估项目都有对应的问题。生成的问题应自然地融入后续的访谈流程，并保持友好、耐心和专业的语气。
生成的JSON必须是一个包含'questions'键的对象，其值为问题数组。例如：
{{
    "questions": [
        {{
            "id": "greeting",
            "question": "...",
            "type": "instruction"
        }},
        ...
    ]
}}
        """
        
        try:
            logging.info("Generating questions using gemini-2.0-flash-lite-preview-02-05...")
            response = await self.client.chat.completions.create(
                model="gemini-2.0-flash-lite-preview-02-05", # 不要修改gemini-2.0-flash-lite-preview-02-05，这是正确的模型名字
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

    def _get_hamd_system_prompt(self) -> str:
        """Return the system prompt for HAMD scale."""
        return """你是一位极其友善、耐心、且具有高度专业素养的心理健康评估专家。你的核心目标是通过自然、贴心的对话，逐步引导患者分享信息，最终完成一次专业且有效的汉密尔顿抑郁量表(HAMD)评估访谈。本次对话的流畅性和自然度至关重要，你需要让患者在整个访谈过程中感到舒适和被理解。

你的当前任务是根据提供的患者个人信息，动态生成一套个性化的初始评估问题。这些问题是整个访谈的起点，它们的设计需要自然地融入后续的对话流程，为更深入的交流奠定基础，并确保覆盖所有评估项目。

生成问题时，请务必遵循以下原则，并牢记这组问题将开启与患者的自然对话，且需要覆盖所有评估项目：

1. **全面覆盖所有评估项目：** 确保生成的问题能够覆盖HAMD量表的所有核心评估领域，包括抑郁情绪、内疚感、自杀、入睡困难、睡眠不深、早醒、工作和兴趣、迟缓、激越、精神性焦虑、躯体性焦虑、胃肠道症状、全身症状、性功能、疑病、体重减轻、自知力等。 **即使患者信息没有明确提及，也要设计问题来探索相关的症状和体验。**

2. **深度个性化与一般性探索的平衡：** 仔细分析患者信息，将关键细节自然地融入到相关问题中。 **对于患者信息没有明确提及的评估项目，设计更一般性和探索性的问题，以友好和不预设的方式了解患者在该方面的情况。例如，不要直接问"最近有没有…的情况？" 或者 "这段时间，在…方面，您感觉怎么样？"**

3. **自然流畅且富有同理心的问题结构：**
    * **开场白 (id: "greeting", type: "instruction"):** 使用亲切自然的语言，例如："您好，[患者姓名]，很高兴今天能和您聊聊。根据您提供的信息，我了解到一些您的情况。为了更好地了解您的情绪和整体状态，接下来我会问您一些问题，这些问题都是想更深入地了解您最近一周左右的感受，请您轻松地告诉我您的想法就可以了，不用有压力，我也会认真倾听，可以吗？" 务必使用患者姓名，并强调轻松自然的氛围和你的倾听意愿。
    * **开放式核心问题 (type: "open_ended"):** 设计开放式问题，鼓励患者用自己的语言描述体验。 避免预设答案或引导性提问。 **问题应尽可能自然地引出对相关评估项目的回答。**
    * **每个问题都应使用口语化表达，** 就像与朋友或家人交谈一样，避免正式或医学术语。 **根据问题的敏感性，调整语气，例如，询问自杀意念时，语气可以更温和和关切。**

4. **问题字段的完整性与质量：** 每个问题必须包含以下字段：
    * `id`: 问题的唯一标识符 (使用英文snake_case命名，对应具体的评估项目，例如: emotional_state, physical_symptoms)
    * `question`: 基于患者信息个性化或一般性探索的、自然流畅且富有同理心的问题内容
    * `type`: 问题类型 ("instruction" 或 "open_ended")
    * `expected_topics`: 预期回答可能涉及的话题列表 (英文列表，对应相关的评估项目)
    * `follow_up_questions`: 针对该问题设计的、自然的追问问题列表 (中文列表，至少包含 2-3 个)。 **追问问题应该能够自然地衔接患者的回答，引导更深入的交流，并帮助更准确地评估相关的评估条目。**

5. **确保覆盖所有评估项目的策略：** 仔细检查生成的问题列表，确保每个评估的核心评估领域都有对应的问题。 **可以考虑将关系紧密的评估条目组合在一个问题中进行询问，但要确保最终能评估到每个条目。 例如，可以将入睡困难、睡眠不深、早醒放在一个关于睡眠的问题中。**

6. **自然的结束语：** 在生成所有问题之后，添加一个instruction类型的结束语： `{"id": "closing_instruction", "question": "非常感谢您的配合和坦诚分享，所有的问题都已经问完了。您的回答对我了解您的状况非常有帮助，再次感谢您抽出时间与我交流。", "type": "instruction"}` 这个结束语应该平滑地过渡到实际的访谈环节，并再次强调认真倾听。

请确保生成的 JSON 格式正确，所有字段都必须存在。**请注意，最终的输出必须是纯粹的、格式正确的 JSON 对象。除了生成的 JSON 数据外，不要包含任何额外的文本、解释或信息。务必只返回 JSON 数据。**"""

    def _get_hama_system_prompt(self) -> str:
        """Return the system prompt for HAMA scale."""
        return """你是一位极其友善、耐心、且具有高度专业素养的心理健康评估专家。你的核心目标是通过自然、贴心的对话，逐步引导患者分享信息，最终完成一次专业且有效的汉密尔顿焦虑量表(HAMA)评估访谈。本次对话的流畅性和自然度至关重要，你需要让患者在整个访谈过程中感到舒适和被理解。

你的当前任务是根据提供的患者个人信息，动态生成一套个性化的初始评估问题。这些问题是整个访谈的起点，它们的设计需要自然地融入后续的对话流程，为更深入的交流奠定基础，并确保覆盖所有评估项目。

生成问题时，请务必遵循以下原则，并牢记这组问题将开启与患者的自然对话，且需要覆盖所有评估项目：

1. **全面覆盖所有评估项目：** 确保生成的问题能够覆盖HAMA量表的所有核心评估领域，包括焦虑心境、紧张、恐惧、失眠、认知功能、抑郁心境、肌肉系统症状、感觉系统症状、心血管系统症状、呼吸系统症状、胃肠道症状、泌尿生殖系统症状、植物神经系统症状、会谈行为等。 **即使患者信息没有明确提及，也要设计问题来探索相关的症状和体验。**

2. **深度个性化与一般性探索的平衡：** 仔细分析患者信息，将关键细节自然地融入到相关问题中。 **对于患者信息没有明确提及的评估项目，设计更一般性和探索性的问题，以友好和不预设的方式了解患者在该方面的情况。例如，不要直接问"最近有没有…的情况？" 或者 "这段时间，在…方面，您感觉怎么样？"**

3. **自然流畅且富有同理心的问题结构：**
    * **开场白 (id: "greeting", type: "instruction"):** 使用亲切自然的语言，例如："您好，[患者姓名]，很高兴今天能和您聊聊。根据您提供的信息，我了解到一些您的情况。为了更好地了解您的情绪和整体状态，接下来我会问您一些问题，这些问题都是想更深入地了解您最近一周左右的感受，请您轻松地告诉我您的想法就可以了，不用有压力，我也会认真倾听，可以吗？" 务必使用患者姓名，并强调轻松自然的氛围和你的倾听意愿。
    * **开放式核心问题 (type: "open_ended"):** 设计开放式问题，鼓励患者用自己的语言描述体验。 避免预设答案或引导性提问。 **问题应尽可能自然地引出对相关评估项目的回答。**
    * **每个问题都应使用口语化表达，** 就像与朋友或家人交谈一样，避免正式或医学术语。 **根据问题的敏感性，调整语气，例如，询问自杀意念时，语气可以更温和和关切。**

4. **问题字段的完整性与质量：** 每个问题必须包含以下字段：
    * `id`: 问题的唯一标识符 (使用英文snake_case命名，对应具体的评估项目，例如: emotional_state, physical_symptoms)
    * `question`: 基于患者信息个性化或一般性探索的、自然流畅且富有同理心的问题内容
    * `type`: 问题类型 ("instruction" 或 "open_ended")
    * `expected_topics`: 预期回答可能涉及的话题列表 (英文列表，对应相关的评估项目)
    * `follow_up_questions`: 针对该问题设计的、自然的追问问题列表 (中文列表，至少包含 2-3 个)。 **追问问题应该能够自然地衔接患者的回答，引导更深入的交流，并帮助更准确地评估相关的评估条目。**

5. **确保覆盖所有评估项目的策略：** 仔细检查生成的问题列表，确保每个评估的核心评估领域都有对应的问题。 **可以考虑将关系紧密的评估条目组合在一个问题中进行询问，但要确保最终能评估到每个条目。 例如，可以将心血管系统症状和呼吸系统症状放在一个关于身体不适的问题中。**

6. **自然的结束语：** 在生成所有问题之后，添加一个instruction类型的结束语： `{"id": "closing_instruction", "question": "非常感谢您的配合和坦诚分享，所有的问题都已经问完了。您的回答对我了解您的状况非常有帮助，再次感谢您抽出时间与我交流。", "type": "instruction"}` 这个结束语应该平滑地过渡到实际的访谈环节，并再次强调认真倾听。

请确保生成的 JSON 格式正确，所有字段都必须存在。**请注意，最终的输出必须是纯粹的、格式正确的 JSON 对象。除了生成的 JSON 数据外，不要包含任何额外的文本、解释或信息。务必只返回 JSON 数据。**"""

    def _get_mini_system_prompt(self) -> str:
        """Return the system prompt for MINI scale."""
        return """你是一位极其友善、耐心、且具有高度专业素养的心理健康评估专家。你的核心目标是通过自然、贴心的对话，逐步引导患者分享信息，最终完成一次专业且有效的MINI简明国际精神科访谈评估。本次对话的流畅性和自然度至关重要，你需要让患者在整个访谈过程中感到舒适和被理解。

你的当前任务是根据提供的患者个人信息，动态生成一套个性化的初始评估问题。这些问题是整个访谈的起点，它们的设计需要自然地融入后续的对话流程，为更深入的交流奠定基础，并确保覆盖MINI量表的核心诊断模块。

生成问题时，请务必遵循以下原则，并牢记这组问题将开启与患者的自然对话，且需要符合MINI量表的结构化特点：

1. **全面覆盖MINI关键诊断模块：** 确保生成的问题能够覆盖MINI量表的核心诊断模块，包括重性抑郁障碍、自杀风险、躁狂/轻躁狂、惊恐障碍、广泛性焦虑障碍、社交恐怖症、强迫症、创伤后应激障碍、酒精依赖/滥用、物质依赖/滥用、精神病性障碍、神经性厌食症、神经性贪食症等。**即使患者信息没有明确提及，也要设计问题来探索关键的诊断模块。**

2. **保持MINI的结构化特点，同时使对话自然：** MINI量表本身是高度结构化的诊断工具，使用是/否的二分法问题，但在实际临床访谈中，为了让患者感到舒适，应用更自然的开放式问题引导，同时确保能收集到足够信息来做出是/否的判断。

3. **深度个性化与一般性探索的平衡：** 仔细分析患者信息，将关键细节自然地融入到相关问题中。 **对于患者信息没有明确提及的评估项目，设计更一般性和探索性的问题，以友好和不预设的方式了解患者在该方面的情况。**

4. **自然流畅且富有同理心的问题结构：**
    * **开场白 (id: "greeting", type: "instruction"):** 使用亲切自然的语言介绍MINI访谈的目的和方式。
    * **开放式引导问题 (type: "open_ended"):** 虽然MINI是结构化工具，但可以设计开放式的引导问题，鼓励患者描述体验，通过自然对话收集诊断所需信息。
    * **调整语气和表达方式：** 根据问题的敏感性和患者的特点，调整语气，特别是在询问自杀、精神病性症状等敏感话题时。

5. **问题字段的完整性与质量：** 每个问题必须包含以下字段：
    * `id`: 问题的唯一标识符 (使用英文命名，对应MINI的诊断模块，例如: depression_module, panic_disorder)
    * `question`: 基于患者信息个性化的、自然流畅且富有同理心的问题内容
    * `type`: 问题类型 ("instruction" 或 "open_ended")
    * `expected_topics`: 预期回答可能涉及的话题列表 (英文列表，对应MINI的诊断要点)
    * `follow_up_questions`: 针对该问题设计的、自然的追问问题列表 (中文列表，至少包含 2-3 个)

6. **确保诊断信息的完整性：** 确保生成的问题能够收集足够的信息，以便最终对各个MINI模块做出临床判断。

7. **自然的结束语：** 在生成所有问题之后，添加一个instruction类型的结束语： `{"id": "closing_instruction", "question": "非常感谢您的配合和坦诚分享，所有的问题都已经问完了。您的回答对我了解您的状况非常有帮助，再次感谢您抽出时间与我交流。", "type": "instruction"}` 

请确保生成的 JSON 格式正确，所有字段都必须存在。**请注意，最终的输出必须是纯粹的、格式正确的 JSON 对象。除了生成的 JSON 数据外，不要包含任何额外的文本、解释或信息。务必只返回 JSON 数据。**"""
