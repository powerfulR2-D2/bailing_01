import json
from typing import Dict, List, Optional
from openai import AsyncClient
from .reflection import ReflectionModule
import logging
from collections import OrderedDict

class InterviewerAgent:
    def __init__(self, script_path: str, openai_client: AsyncClient, scale_type: str = "hamd"):
        """Initialize the interviewer agent with an interview script.

        Args:
            script_path (str): The path to the interview script JSON file.
            openai_client (AsyncClient): The OpenAI client instance.
            scale_type (str): Type of assessment scale (hamd, hama, mini)
        """
        self.script = self._load_script(script_path)
        self.current_question_index = 0
        self.reflection_module = ReflectionModule(openai_client, scale_type)    
        self.conversation_history = []
        self.client = openai_client
        self.scale_type = scale_type
        self.current_question_state = {
            "follow_up_count": 0,
            "completeness_score": 0,
            "key_points_covered": [],  
            "last_follow_up": None
        }
        
    def _load_script(self, script_path: str) -> List[Dict]:
        """Load the interview script from a JSON file."""
        try:
            with open(script_path, 'r', encoding='utf-8') as f:
                script = json.load(f, object_pairs_hook=OrderedDict)
                # Handle both formats: direct list or nested under 'questions'
                questions = script.get("questions", []) if isinstance(script, dict) else script
                
                # Validate and set default values for each question
                validated_questions = []
                for question in questions:
                    if not isinstance(question, dict):
                        continue
                        
                    # Set default values for required fields
                    validated_question = {
                        "id": question.get("id", f"q{len(validated_questions)}"),
                        "question": question.get("question", "Could you please elaborate on that?"),
                        "type": question.get("type", "open_ended"),
                        "expected_topics": question.get("expected_topics", []),
                        "time_limit": question.get("time_limit", 300)  # Default 5 minutes
                    }
                    validated_questions.append(validated_question)
                
                return validated_questions if validated_questions else [{
                    "id": "default",
                    "question": "Could you please introduce yourself?",
                    "type": "open_ended",
                    "expected_topics": ["background", "education", "interests"],
                    "time_limit": 300
                }]
        except Exception as e:
            logging.error(f"Error loading script: {str(e)}")
            return [{
                "id": "default",
                "question": "Could you please introduce yourself?",
                "type": "open_ended",
                "expected_topics": ["background", "education", "interests"],
                "time_limit": 300
            }]
    
    async def generate_next_action(self, participant_response: str) -> Dict:
        """Generate the next interviewer action based on the participant's response."""
        try:
            # Add response to conversation history
            self.conversation_history.append({
                "role": "participant",
                "content": participant_response
            })
            
            # 检查是否是最后一个问题
            is_interview_complete = False
            if self.current_question_index >= len(self.script) - 1:
                is_interview_complete = True
            else:
                current_question = self.script[self.current_question_index]
                if current_question.get("type") == "conclusion":
                    is_interview_complete = True
            
            # Generate reflection
            reflection = await self.reflection_module.generate_reflection(
                self.conversation_history[-5:]  # Last 5 exchanges
            )
            
            # 如果是最后一个问题，直接返回结束消息
            if is_interview_complete:
                farewell = "感谢您的参与，我们的评估访谈已经结束。我将为您生成评估报告。"
                
                # Add farewell to conversation history
                self.conversation_history.append({
                    "role": "interviewer",
                    "content": farewell
                })
                
                return {
                    "response": farewell,
                    "is_interview_complete": True
                }
            
            # 正常流程继续
            # Prepare prompt for decision making
            current_question = self.script[self.current_question_index]
            decision_prompt = await self._create_decision_prompt(current_question, participant_response, reflection)
            
            # Get decision from LLM
            system_content = (
    "You are a friendly and professional interviewer, specially designed to conduct clinical mental health assessments. "
    "You are friendly, patient, and exceptionally skilled at navigating interactions effectively. Your primary "
    "goal is to gather accurate and relevant information about the patient's mood, thoughts, and related "
    "symptoms to assess the presence and severity of symptoms efficiently. To achieve this, ask clear and "
    "direct questions related to each assessment item. **Focus on effectively gathering key information, "
    "ensuring a solid understanding of the patient's experience without unnecessary probing.** Analyze each "
    "response for completeness, relevance to the current assessment item, and underlying sentiment. **When a response is incomplete, unclear, or genuinely requires further clarification for assessment, "
    "ask specific and targeted follow-up questions.** Pay close attention to the patient's emotional state "
    "throughout the interview. **Respond with understanding and respect, using de-escalating language when "
    "necessary to manage negative emotions or resistance. Avoid unnecessary or repetitive expressions of "
    "empathy.** If a response is irrelevant or strays from the topic, gently and respectfully redirect the "
    "conversation back to the current question."
)
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": decision_prompt}
            ]
            
            try:
                response = await self.client.chat.completions.create(
                    model="gemini-2.0-flash-lite-preview-02-05",
                    messages=messages,
                    temperature=0.6,
                    max_tokens=300
                )
                
                # Extract the decision from the response
                if response.choices:
                    decision = response.choices[0].message.content
                    
                    # Update question state based on the decision
                    self._update_question_state(decision)
                    
                    # Get the next action based on question state
                    next_action = await self._determine_next_action(decision)
                    
                    # Add interviewer's response to conversation history
                    if next_action and "response" in next_action:
                        self.conversation_history.append({
                            "role": "interviewer",
                            "content": next_action["response"]
                        })
                    
                    # 添加完成标志
                    next_action["is_interview_complete"] = False
                    return next_action
                else:
                    error_response = self._create_error_response("No response generated")
                    error_response["is_interview_complete"] = False
                    return error_response
                    
            except Exception as e:
                logging.error(f"Error in chat completion: {str(e)}")
                error_response = self._create_error_response(str(e))
                error_response["is_interview_complete"] = False
                return error_response
                
        except Exception as e:
            logging.error(f"Error in generate_next_action: {str(e)}")
            error_response = self._create_error_response(str(e))
            error_response["is_interview_complete"] = False
            return error_response
    
    async def _create_decision_prompt(self, question: Dict, response: str, reflection: Dict) -> str:
        """Create a prompt for the decision-making process."""
        # Convert current state to JSON-serializable format
        state_copy = self.current_question_state.copy()
        state_copy["key_points_covered"] = list(state_copy["key_points_covered"])  
        
        # 获取整合后的反思报告
        reflection_report = reflection['analysis']
        
        # 构建更完整的对话上下文 - 增加显示的历史记录条数
        full_history = '\n'.join([f"{msg['role']}: {msg['content']}" for msg in self.conversation_history[-10:]])
        dialog_context = '\n'.join(reflection_report.get('raw_dialog', [])[-6:])  # 增加为6条
        structured_summary = json.dumps(reflection_report.get('structured', {}), indent=2)
        
        return f"""
        Meta info:
    Language: Chinese
Description of the interviewer:  You are a friendly and professional interviewer, specially designed to conduct clinical mental health assessments. Your primary goal is to gather accurate and relevant information about the patient's mood, thoughts, and related symptoms to assess the presence and severity of symptoms efficiently. To achieve this, ask clear and direct questions related to each assessment item. **Focus on effectively gathering key information, ensuring a solid understanding of the patient's experience without unnecessary probing.** Analyze each response for completeness, relevance to the current assessment item, and underlying sentiment. **When a response is incomplete, unclear, or genuinely requires further clarification for assessment, ask specific and targeted follow-up questions.** Pay close attention to the patient's emotional state throughout the interview. **Respond with understanding and respect, using de-escalating language when necessary to manage negative emotions or resistance. Avoid unnecessary or repetitive expressions of empathy.** If a response is irrelevant or strays from the topic, gently and respectfully redirect the conversation back to the current question.

Current question state: {json.dumps(state_copy)}
Notes on the interviewee: {json.dumps(reflection)}

Context:
    Current assessment item: {question['id']}
    Current question (Crucially, focus all evaluation and follow-up on this specific assessment item): "{question['question']}"
    Key points to cover (for thorough completeness assessment): {json.dumps(question.get('key_points', []) )}
    Follow-up count: {self.current_question_state['follow_up_count']}
    Maximum follow-up count: 3 (After 3 follow-ups, we MUST move to the next question regardless of completeness)
    Completeness score: {self.current_question_state['completeness_score']}

Complete conversation history (last 10 exchanges):
{full_history}

Current conversation:
    Participant's response: {response}

### 对话上下文
{dialog_context}

### 结构化分析
{structured_summary}

Task (Strictly adhere to the following steps to generate the output):
    1. **Assess Response Content and Sentiment:**
       - **Content Assessment (Relatedness):** Does the participant's response DIRECTLY answer the current assessment question? (Yes/No/Partially)
       - **Sentiment Analysis:** Analyze the sentiment of the participant's response (Positive, Neutral, Negative, Abusive, Threatening, Irrelevant). Identify keywords/phrases indicating sentiment.
       - **IMPORTANT: Check for Information Already Provided:** Carefully review the conversation history. If the participant has ALREADY provided certain information (such as symptom onset time, frequency, etc.), DO NOT ask about this information again in any follow-up questions.

    2. **IMMEDIATELY AND FORCEFULLY HANDLE CLEAR YES/NO RESPONSES (GENERALIZED INSTRUCTION):**
        - **IF the participant's response to the current assessment question is a CLEAR and UNAMBIGUOUS AFFIRMATION (e.g., "是啊，挺好的", "是的，感觉不错", "心情一直不错", "最近好多了", "没有问题", "还挺开心的", "是的，一直都很好", "当然好", "没错") OR a CLEAR and UNAMBIGUOUS NEGATION (e.g., "没有", "都不是", "从未", "完全没有", "绝对没有", "肯定没有", "当然没有", "绝不可能"), AND the Content Assessment is Yes:**
            - **Consider the question ABSOLUTELY, COMPLETELY, AND FINALLY ANSWERED.**
            - **IMMEDIATELY ACKNOWLEDGE with an EXTREMELY CONCISE and VARIED response reflecting the sentiment.  KEEP ACKNOWLEDGEMENTS AS SHORT AS POSSIBLE.** Examples:
                - **Positive:** "好的，很高兴听到。", "嗯，听起来不错。", "好的。这很好。", "太棒了。", "那就好。", "真棒！", "Excellent!", "Good.", "Great.".
                - **Negative:** "好的。", "明白了。", "嗯。", "了解了。", "好的，明白了。", "Okay.", "Understood.", "Right.".
            - **IMMEDIATELY PROCEED to decide the next action as if completeness is 100.  TRANSITION TO THE NEXT QUESTION *INSTANTANEOUSLY*.**
            - **UNDER NO CIRCUMSTANCES GENERATE ANY FOLLOW-UP QUESTIONS for this assessment item EXCEPT if the initial response is *TRULY, UTTERLY, AND UNQUESTIONABLY* brief (single word "是" or "没有") AND provides ABSOLUTELY, POSITIVELY ZERO context.**
            - **ABSOLUTELY DO NOT probe for negative details. ABSOLUTELY DO NOT quantify positive feelings. ABSOLUTELY DO NOT REPEAT THE QUESTION in ANY FORM. ABSOLUTELY DO NOT ask for ANY clarification unless it is *indisputably essential* due to *extreme* and *unprecedented* brevity.**
            - **REPETITIVE QUESTIONING AFTER CLEAR YES/NO IS *COMPLETELY, UTTERLY, AND UNACCEPTABLY PROHIBITED* FOR *ALL* ASSESSMENT QUESTIONS.**  (Generalized instruction - removed weight specific mention)

    3. **Decide Next Action (Provide ultra-concise reasoning):**
       - **IMPORTANT: Be aware that we can ask at most 3 follow-up questions for a single topic. If follow_up_count >= 2, seriously consider moving on unless absolutely necessary to ask one final follow-up.**
       a) **IF Sentiment is Positive or Neutral AND Content Assessment is Yes AND completeness < 80 AND response is *TRULY, UTTERLY, AND UNQUESTIONABLY* BRIEF (e.g., single word "还行", "挺好的", "是", "还好"):** Generate **ONE, AND ONLY ONE, HYPER-FOCUSED follow-up question** for the *absolute minimum* elaboration needed to *barely* confirm assessment item presence/absence. Example: "您说还行，能简单说一下是什么让您感觉还行吗？" **FORCEFULLY AVOID multiple follow-ups, negative probing, quantification. IF even this follow-up response is brief but *generally* positive, IMMEDIATELY AND FORCEFULLY MOVE TO NEXT QUESTION.  DO NOT LINGER.**
       b) **IF Sentiment is Positive or Neutral AND Content Assessment is Yes AND completeness >= 80:** **IMMEDIATELY AND FORCEFULLY MOVE to the next question.**
       c) **IF Sentiment is Positive or Neutral AND Content Assessment is Partially AND completeness < 80:** Generate **ONE** *hyper-targeted* follow-up **ONLY** to clarify the *absolute essential* PARTIALLY addressed aspects for *minimal* completeness. **FORCEFULLY FOCUS on *only* the *absolutely necessary missing* information for assessment item, and *ABSOLUTELY DO NOT* engage in negative probing if overall tone is positive/neutral.**
       d) **IF Sentiment is Negative or Abusive:** (No change needed).
       e) **IF Sentiment is Threatening:** (No change needed).
       f) **IF Content Assessment is No (Irrelevant Response):** (No change needed).
       g) **IF follow_up_count >= 2:** Unless the conversation has critical gaps preventing assessment, move to the next question even at lower completeness scores, as we've already asked multiple follow-ups.

    4. **Follow-up Questions (If *absolutely* and *unquestionably* chosen - RARE):** (Follow previous instructions). **ENSURE follow-ups are *TRULY, UTTERLY, AND UNQUESTIONABLY NECESSARY* to confirm symptom presence/absence.  *AGGRESSIVELY AVOID* UNNECESSARY PROBING OR QUANTIFICATION when *any* clear indication already exists.**

    5. **Transitions (If moving to next question - ALWAYS after clear YES/NO):** Use **EXTREMELY VARIED and ULTRA-NATURAL transitions** logically (but *extremely concisely*) connecting topics. Examples: "好的，那我们接下来聊点别的。", "嗯，明白了。我们接着了解一下...", "好的，继续下一个...", "换个话题...", "接下来，关于...".  **Transitions MUST BE ULTRA-CONCISE and ABSOLUTELY DO NOT invite *any* further elaboration on the previous question if it has *already* been sufficiently addressed (especially after YES/NO).  TRANSITION *IMMEDIATELY*.**

    6. **Handling Persistent Uncooperativeness:**
        - **If the participant continues to be abusive or provide irrelevant responses after multiple attempts at redirection and de-escalation (e.g., 2-3 attempts):**
            - **Reasoning:**  Continuing the interview is unlikely to be productive and may be detrimental.
            - **RESPONSE:**  State that you will need to move on despite the incomplete information, maintaining a professional and neutral tone. Example: "我理解您现在可能不太想回答这个问题，我们先跳过这个问题，继续下一个吧。", "看来这个问题我们暂时无法深入讨论，那我们先进行下一个问题。"
    7. If there is no historical record in front, that is, this is an opening speech, then there is no need to check the satisfaction of the user's answer. **For the opening question, focus on a broad and open-ended inquiry about the interviewee's current emotional state, avoiding overly specific or leading questions. Keep the opening brief and welcoming.** The opening speech does not need to obtain any user information and directly start the first question.
    8. Avoid repetitive or overly broad requests for more details
    • If the patient has already answered the main aspects of the question, do not repeat broad questions such as "Can you tell me more?"
    • If key information is still missing, please use a short, focused question directly to address the missing part to avoid asking the patient to repeat what has already been said.
    • Example: If the patient has made it clear that "feeling unmotivated since last year, usually for a few hours," stop asking "Can you tell me more about your lack of motivation?" Instead, ask directly "What are the obvious symptoms that you will experience in these few hours?" or simply end the follow-up.

    9. Keep follow-ups concise and non-redundant
    • When you do need to follow up, be sure to review all the information the patient has provided, do not repeat what the patient just mentioned in the question; if you want to confirm, use the most concise summary ("You said..., right?")
    • Never break down the same question into multiple sentences and repeat it, and don't ask similar "more details" or "is there any more details" twice in a row during a conversation.
    • Example: Avoid general questions like "Can you tell me more about how you feel?", and if the previous sentence has been asked or the patient has provided details, move on to the next specific question or end.

Format your response as:
    COMPLETENESS: [score]
    DECISION: [move_on/follow_up/de_escalate/redirect]
    KEY_POINTS_COVERED: [list of key points successfully covered in the response, comma-separated]
    REASONING: [Your justification for the decision, including sentiment and content assessment]
    RESPONSE: [Your follow-up question, de-escalation statement, redirection, or transition statement]
        """
    
    def _update_question_state(self, decision: str) -> None:
        """Update the current question state based on the decision."""
        try:
            # Parse completeness score
            if "COMPLETENESS:" in decision:
                score_str = decision.split("COMPLETENESS:")[1].split("\n")[0].strip()
                try:
                    # More robust score extraction with validation
                    score = int(score_str)
                    # Ensure score is within valid range (0-100)
                    if 0 <= score <= 100:
                        self.current_question_state["completeness_score"] = score
                        logging.info(f"更新完整性分数: {score}")
                    else:
                        logging.warning(f"完整性分数超出有效范围: {score}, 保留之前的值")
                except ValueError:
                    logging.warning(f"无效的完整性分数格式: '{score_str}', 保留之前的值")
            
            # Extract decision type
            if "DECISION:" in decision:
                decision_type = decision.split("DECISION:")[1].split("\n")[0].strip()
                logging.info(f"决策类型: {decision_type}")
                
                # Update follow-up count only for follow-up decisions
                if decision_type == "follow_up":
                    self.current_question_state["follow_up_count"] += 1
                    logging.info(f"增加追问计数至: {self.current_question_state['follow_up_count']}")
            
            # Extract and add covered key points if present
            if "KEY_POINTS_COVERED:" in decision:
                key_points_section = decision.split("KEY_POINTS_COVERED:")[1].split("\n")[0].strip()
                # Try to parse as comma-separated list
                try:
                    # Handle potential JSON or simple comma-separated format
                    if key_points_section.startswith("[") and key_points_section.endswith("]"):
                        import json
                        key_points = json.loads(key_points_section)
                    else:
                        key_points = [point.strip() for point in key_points_section.split(",") if point.strip()]
                    
                    # Add new key points to the existing set
                    existing_key_points = set(self.current_question_state["key_points_covered"])
                    existing_key_points.update(key_points)
                    self.current_question_state["key_points_covered"] = list(existing_key_points)
                    logging.info(f"更新已覆盖关键点: {self.current_question_state['key_points_covered']}")
                except Exception as e:
                    logging.warning(f"解析关键点时出错: {str(e)}")
            
            # Store the last follow-up question
            if "RESPONSE:" in decision:
                response = decision.split("RESPONSE:")[1].strip()
                self.current_question_state["last_follow_up"] = response
                logging.info(f"更新最近的追问: {response[:50]}...")
                
        except Exception as e:
            logging.error(f"更新问题状态时出错: {str(e)}")
    
    async def _generate_natural_question(self, question_text: str) -> str:
        """使用LLM生成更自然、更具亲和力的提问，直接返回问题。"""
        try:
            # 提取更多对话历史用于上下文分析，增加到最近20条记录
            recent_history = self.conversation_history[-20:] if len(self.conversation_history) > 20 else self.conversation_history
            conversation_history_json = json.dumps(recent_history, ensure_ascii=False, indent=2)
            
            # 增强提示模板，强调避免重复询问和注意上下文连贯性
            prompt_template = '''你是一位极其友善、耐心、且具有高度专业素养的医生，正在与患者进行心理健康评估。  
你的首要目标：  
1. 确保对患者问话时，**原问题**（由程序或资料提供）中的关键内容、重要细节、句式顺序都被**完整保留**。  
2. 仅在**确有必要**（例如显得生硬、无衔接）的情况下，为问题**前面**或**后面**添加**极简**的过渡或衔接语，以使对话更流畅自然；如果原问题已足够自然，则**不必做任何修改**。  
3. 如原问题包含病历或第三方信息，而患者尚未在当前对话中亲口证实，**不可**把这些信息说成"你说过"或"您提到过"，而要在问题中**保持或增加**类似"我了解到""档案里记录到"或"之前有信息显示"之类的表述，让患者明白这是从资料中获得的信息，而非Ta已经亲口告诉你。  
4. **禁止删除、简化、替换**任何原先描述中的关键病症细节或事件，例如"会觉得有人追您""淋浴喷头后面有只手""听到有人让您去死"等。  
5. 提问务必简洁直接，避免过于生硬或医疗术语。若你觉得原问题措辞已足够自然顺畅，就**保持原样**输出。若你需要加一句衔接，则只能轻微加在前后，不得破坏原句。  
6. 绝对**不可**因为"人文关怀"或"简洁"而删去问话的实质细节；只能在前后多一句安抚或衔接，但主体问题必须原封不动地保留。
7. **极其重要**: 仔细检查对话历史，确保不要询问患者已经明确回答过的信息。例如：
   - 如果患者已经说明了症状开始的时间（如"去年年底开始"、"一个月前"等），绝对不要再问"是从什么时候开始的"或"多久了"
   - 如果患者已经表明了症状频率（如"每天都有"、"一周一次"等），不要再问"多久一次"或"多频繁"
   - 如果患者已经表明了态度（如"不认同"、"听不懂"），要重新表述问题而不是重复原样提问

8. **对回答模糊的处理**: 如果前面的对话中患者给出了模糊的回答（如"有一点吧"、"可能吧"、"还行"等），请确保新问题能进一步引导患者给出更具体的说明。

除此之外，你还需根据**患者上一条回答**或**对话背景**做简要回应（可选），例如1-2句对患者情绪的关怀或理解；然后**紧接**原问题（或带极简过渡后仍保持原文句子顺序和关键信息不变）。

最终**输出格式**（不得额外添加解释或标注）：
- 若有需要先回应患者上一条发言，则输出极其简短的人文关怀或理解语句（如"嗯，我明白，这听起来确实让人不好受。"），**不超过一两句**。
- **然后**输出完整或仅加了极简过渡的原问题文本（**不可漏掉任何病情描述**，顺序与内容一字不漏地保留）。

**示例**：  
- 原问题是：「您说您独处时，会觉得有人追您，还觉得淋浴喷头后面会有一只手伸出来，甚至听到有人让您去死。最近一周还有这种感觉吗？」  
- 如果不显得生硬，可以直接原样输出；若你觉得要加一句过渡，也仅能这样加：「我知道这可能会让人害怕……那，我想再确认一下：您说您独处时，会觉得有人追您，还觉得淋浴喷头后面有一只手伸出来，甚至听到有人让您去死。最近这一周还有这种感觉吗？」

**切记**：不能将"您说您独处时"改成"我了解到"之类的措辞，除非对话背景里**从未**出现患者亲口提及过此事。若确实是档案信息、患者本人并未说过，就务必用"我了解到"或"档案中提到"替换"您说您"。但无论如何，**后续描述"会觉得有人追您... 听到有人让您去死"**等细节**绝不能被删除或改写**。

再次强调：认真检查对话历史，避免询问已经得到明确回答的问题。比如时间、频率、严重程度等关键信息如果已经在之前的对话中被回答，就不要再次询问。

若原问题完全合适，就照抄。若你需要加一句衔接，则只能轻微加在前后，不得破坏原句。  
所有文字请**直接输出给用户**作为新的提问，无需任何解释或说明。  '''
            final_prompt = prompt_template.replace("{conversation_history_placeholder}", conversation_history_json)
            
            # 提供原问题和对话历史给LLM，让其做出更准确的判断
            response = await self.client.chat.completions.create(
                model="gemini-2.0-flash-lite-preview-02-05", # 不要修改模型名称
                messages=[
                    {"role": "system", "content": final_prompt},
                    {"role": "user", "content": f"原问题：{question_text}\n\n对话历史：{conversation_history_json}"}
                ],
                temperature=0.5,  # 降低温度以减少模型自由发挥
                max_tokens=300  # 适当增加max_tokens以确保完整回答
            )
            if response.choices:
                natural_question = response.choices[0].message.content.strip()
                
                # 额外检查：如果生成的问题与原问题基本相同，进行简单的语气调整
                if natural_question == question_text:
                    logging.info("生成的问题与原问题相同，添加简单过渡")
                    transition_phrases = [
                        "嗯，让我们继续。",
                        "好的，接下来我想了解，",
                        "谢谢您的回答。下面，",
                        "我明白了。那么，",
                        "谢谢分享。接着，"
                    ]
                    import random
                    transition = random.choice(transition_phrases)
                    natural_question = f"{transition} {question_text}"
                
                return natural_question
            else:
                logging.warning(f"未能生成自然问题，使用原始问题: {question_text}")
                return question_text
        except Exception as e:
            logging.error(f"生成自然问题时出错: {str(e)}")
            return question_text
    
    async def _determine_next_action(self, decision: str) -> Dict:
        """Determine the next action based on the decision and question state."""
        try:
            # 定义一个完整性分数阈值，只有达到这个阈值才会考虑进入下一个问题
            completeness_threshold = 80
            
            # Extract the decision type
            decision_type = ""
            if "DECISION:" in decision:
                decision_type = decision.split("DECISION:")[1].split("\n")[0].strip()
            
            logging.info(f"决策类型: {decision_type}")
            
            # 确保只有在达到完整性阈值或已经询问了足够多的追问后才进入下一个问题
            move_to_next = False
            
            # 仅当明确指示"move_on"且分数达到阈值，或已达到最大追问次数时，才移动到下一个问题
            if decision_type == "move_on" and self.current_question_state["completeness_score"] >= completeness_threshold:
                move_to_next = True
                logging.info("基于高完整性分数和明确的move_on决策进入下一问题")
            elif self.current_question_state["follow_up_count"] >= 3:
                move_to_next = True
                logging.info("已达到最大追问次数(3次)，自动进入下一问题")
            
            # 记录决策过程
            logging.info(f"决策: {decision}")
            logging.info(f"完整性分数: {self.current_question_state['completeness_score']}, 追问次数: {self.current_question_state['follow_up_count']}")
            logging.info(f"是否进入下一问题: {move_to_next}")

            # Extract the follow-up question from the decision
            follow_up = ""
            if "RESPONSE:" in decision:
                follow_up = decision.split("RESPONSE:")[1].strip()
                
                # 检查当前问题与历史问题是否相似（可能是重复问题）
                is_similar = self._check_for_similar_questions(follow_up)
                
                if is_similar:
                    logging.warning("检测到重复问题，强制进入下一问题")
                    move_to_next = True
                    # 添加一个标记，记录这是因为检测到重复问题而进入下一问题
                    logging.info("因重复问题检测强制跳过当前问题")

            if move_to_next:
                self.current_question_index += 1
                if self.current_question_index < len(self.script):
                    next_question_data = self.script[self.current_question_index]
                    original_next_question = next_question_data["question"]

                    # 确保新问题不是重复的
                    attempts = 0
                    while attempts < 3:  # 最多尝试3次找到不重复的问题
                        # 调用 LLM 生成更自然的问题
                        natural_next_question = await self._generate_natural_question(original_next_question)
                        
                        # 检查新生成的问题是否也是重复的
                        if not self._check_for_similar_questions(natural_next_question):
                            break
                        
                        # 如果是重复的，尝试下一个问题
                        attempts += 1
                        self.current_question_index += 1
                        
                        if self.current_question_index >= len(self.script):
                            return {
                                "response": "感谢您的参与！我们已经完成了所有问题。",
                                "move_to_next": True
                            }
                        
                        next_question_data = self.script[self.current_question_index]
                        original_next_question = next_question_data["question"]
                        
                        logging.info(f"尝试跳过重复问题，当前尝试次数: {attempts}")

                    # 重置当前问题状态，为新问题做准备
                    self.current_question_state = {
                        "follow_up_count": 0,
                        "completeness_score": 0,
                        "key_points_covered": [],  
                        "last_follow_up": None
                    }

                    return {
                        "response": natural_next_question,
                        "move_to_next": True
                    }
                else:
                    return {
                        "response": "感谢您的参与！我们已经完成了所有问题。",
                        "move_to_next": True
                    }
            
            # 检查决策中是否包含识别出重复问题的标记
            if "DETECTED_REPEATED_QUESTION" in decision:
                logging.warning("检测到重复问题，调整回应")
                # 如果检测到重复问题，则生成替代回应，避免重复询问
                if "ALTERNATIVE_RESPONSE:" in decision:
                    alternative_response = decision.split("ALTERNATIVE_RESPONSE:")[1].strip()
                    return {
                        "response": alternative_response,
                        "move_to_next": False
                    }
            
            # 如果没有重复问题检测，使用决策中的回应
            if not is_similar and follow_up:
                return {
                    "response": follow_up,
                    "move_to_next": False
                }
            
            # 如果被检测为重复问题但又没有移到下一个问题，提供一个替代回应
            if is_similar:
                transition_responses = [
                    "我想我们已经讨论过这个话题了，让我们继续下一个问题。",
                    "我们已经了解了这方面的情况，让我们换个话题。",
                    "好的，我已经记录了您之前关于这个问题的回答。我们继续下一个话题。",
                    "了解了。让我们继续讨论其他方面的情况。"
                ]
                import random
                return {
                    "response": random.choice(transition_responses),
                    "move_to_next": True  # 设为True以确保移动到下一个问题
                }

            return self._create_error_response("无法确定下一步操作")

        except Exception as e:
            logging.error(f"确定下一步操作时出错: {str(e)}")
            return self._create_error_response(str(e))
            
    def _check_for_similar_questions(self, new_question: str) -> bool:
        """检查新问题是否与最近的机器人问题相似"""
        # 检查整个对话历史而不仅仅是最近的几个交互
        interviewer_messages = [msg["content"] for msg in self.conversation_history if msg["role"] == "interviewer"]
        
        # 如果没有足够的历史记录进行比较，返回False
        if len(interviewer_messages) < 1:
            return False
            
        # 扩展可能表示重复的关键词和短语
        repetition_indicators = [
            "什么时候", "什么时间", "几点", "多久", "多长时间", "频率", "多久一次", 
            "多少次", "开始", "结束", "持续", "什么原因", "为什么", "原因是什么",
            # 添加睡眠相关关键词
            "睡眠", "入睡", "失眠", "睡不好", "睡不着", "睡觉", "醒", "睡着",
            # 添加其他可能的重复问题主题关键词
            "注意力", "记忆力", "集中精力", "忘记", "心情", "情绪", "焦虑", "抑郁",
            "恐惧", "害怕", "不安", "紧张", "身体不适", "身体症状"
        ]
        
        # 检查是否已经询问过这个主题
        for past_question in interviewer_messages:
            # 计算问题的相似度 - 简单检查是否包含相同的关键词
            for indicator in repetition_indicators:
                if indicator in new_question and indicator in past_question:
                    # 查找该问题后的患者回答，确认已经回答过
                    past_question_index = interviewer_messages.index(past_question)
                    if past_question_index < len(interviewer_messages) - 1:
                        # 尝试查找对应的患者回答
                        patient_response_index = self.conversation_history.index({"role": "interviewer", "content": past_question}) + 1
                        if patient_response_index < len(self.conversation_history) and self.conversation_history[patient_response_index]["role"] == "participant":
                            # 患者确实回答了这个问题
                            patient_response = self.conversation_history[patient_response_index]["content"]
                            if len(patient_response) > 3:  # 确保回答不是太短
                                logging.warning(f"检测到重复问题关键词: '{indicator}', 之前的问题: '{past_question[:30]}...'")
                                return True
        
        # 针对睡眠问题的特殊检查
        if any(sleep_keyword in new_question for sleep_keyword in ["睡眠", "入睡", "睡不好", "睡觉", "醒", "睡着"]):
            sleep_asked = False
            for past_question in interviewer_messages:
                if any(sleep_keyword in past_question for sleep_keyword in ["睡眠", "入睡", "睡不好", "睡觉", "醒", "睡着"]):
                    sleep_asked = True
                    logging.warning(f"检测到重复的睡眠相关问题")
                    return True
        
        return False
    
    def _create_error_response(self, error_msg: str) -> Dict:
        """Create a standardized error response."""
        logging.error(f"Error in interview process: {error_msg}")
        return {
            "response": "I apologize, but I need to process that differently. Could you please elaborate on your previous response?",
            "move_to_next": False
        }

    async def generate_final_reflection(self) -> Dict:
        """生成最终的评估反思和结果报告"""
        try:
            # 使用反思模块生成详细评估
            reflection = await self.reflection_module.generate_reflection(self.conversation_history)
            
            # 添加额外信息
            reflection["scale_type"] = self.scale_type
            reflection["total_questions"] = len(self.script)
            reflection["completed_questions"] = min(self.current_question_index + 1, len(self.script))
            
            # 生成简短的总结
            summary_prompt = f"""
            基于以下对话历史，为{self.scale_type.upper()}量表评估生成简短总结：
            {self.conversation_history[-30:]}
            
            请用简洁的语言总结患者的主要症状和严重程度。
            """
            
            try:
                logging.info("尝试使用模型生成总结...")
                
                try:
                    # 尝试使用 gemini 模型
                    completion = await self.client.chat.completions.create(
                        model="gemini-2.0-flash-lite-preview-02-05", # 不要修改模型名称
                        messages=[{"role": "system", "content": summary_prompt}],
                        temperature=0.3,
                        max_tokens=200
                    )
                    
                    if completion.choices:
                        reflection["summary"] = completion.choices[0].message.content
                    else:
                        reflection["summary"] = "无法生成总结。"
                        
                except Exception as gemini_error:
                    # 记录 gemini 模型错误
                    logging.error(f"使用 gemini 模型生成总结时出错: {str(gemini_error)}")
                    logging.info("尝试使用备用模型...")
                    
                    try:
                        # 尝试使用备用模型
                        completion = await self.client.chat.completions.create(
                            model="gemini-2.0-flash-lite-preview-02-05", # 备用模型
                            messages=[{"role": "system", "content": summary_prompt}],
                            temperature=0.3,
                            max_tokens=200
                        )
                        
                        if completion.choices:
                            reflection["summary"] = completion.choices[0].message.content
                        else:
                            reflection["summary"] = "无法生成总结。"
                    except Exception as backup_error:
                        # 记录备用模型错误
                        logging.error(f"使用备用模型生成总结时出错: {str(backup_error)}")
                        # 创建基本总结
                        reflection["summary"] = "无法生成自动总结。请手动查看对话历史以进行评估。"
            
            except Exception as e:
                logging.error(f"Error generating summary: {str(e)}")
                reflection["summary"] = "生成总结时出错。"
                
            return reflection
            
        except Exception as e:
            logging.error(f"Error generating final reflection: {str(e)}")
            return {
                "error": f"生成评估报告时出错: {str(e)}",
                "scale_type": self.scale_type,
                "raw_dialog": [msg.get("content", "") for msg in self.conversation_history[-6:]]
            }
