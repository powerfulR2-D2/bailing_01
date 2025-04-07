import json
import logging
from collections import OrderedDict
from typing import Any, Dict, List, Optional

import httpx
# Use alias for clarity when creating instances later
from openai import AsyncClient as OpenAIAsyncClient

# Configure logging at the module level (or application entry point)
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__) # Using __name__ is standard practice

class InterviewerAgent:
    # def __init__(self, script_path: str, openai_client: AsyncClient, scale_type: str = "hamd"): # Old signature
    def __init__(self, script_path: str, llm_config: Dict, scale_type: str = "hamd"): # New signature
        """Initialize the interviewer agent with an interview script and LLM config.

        Args:
            script_path (str): The path to the interview script JSON file.
            llm_config (Dict): Configuration for the LLM provider (openai or ollama).
                               Expected structure: {"llm": {"provider": "...", "openai": {...}, "ollama": {...}}}
            scale_type (str): Type of assessment scale (hamd, hama, mini)
        """
        self.script = self._load_script(script_path)
        self.current_question_index = 0
        self.conversation_history = []
        # self.client = openai_client # Removed
        self.llm_config = llm_config # Added
        self.scale_type = scale_type
        self.current_question_state = {
            "follow_up_count": 0,
            "completeness_score": 0,
            "key_points_covered": [],
            "last_follow_up": None
        }
        # Initialize shared HTTP client for Ollama and potentially others
        self._http_client = httpx.AsyncClient(verify=True, timeout=60.0) # Added, verify=True initially

        # Optional: Pre-initialize OpenAI client if provider is openai
        self._openai_client_instance: Optional[OpenAIAsyncClient] = None # Added
        if self.llm_config.get("llm", {}).get("provider") == "openai":
            openai_conf = self.llm_config.get("llm", {}).get("openai", {})
            api_key = openai_conf.get("api_key")
            base_url = openai_conf.get("base_url")
            if api_key: # Only initialize if api_key is provided
                self._openai_client_instance = OpenAIAsyncClient(
                    api_key=api_key,
                    base_url=base_url # base_url can be None, OpenAI client handles default
                )
                logging.info("Pre-initialized OpenAI client.")
            else:
                logging.warning("OpenAI provider selected, but no API key found in llm_config. Direct calls will fail unless a key is provided later or implicitly.")


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
            
            # 生成对话分析和反思
            # 最新的几条对话
            recent_history = self.conversation_history[-15:] if len(self.conversation_history) > 15 else self.conversation_history
            dialog_snippets = [f"{msg.get('role', 'unknown')}: {msg.get('content', '')}" for msg in recent_history if msg.get('content')]
            
            # 将反思分析作为prompt的一部分，而不是单独的LLM调用
            reflection_analysis_prompt = """
首先，请分析以下对话历史：

{history}

分析要点：
1. 患者提到了哪些症状？(列出1-5个主要症状)
2. 患者提到了哪些时间信息？(症状开始、持续时间等)
3. 还有哪些需要澄清的信息？(哪些关键细节仍不清楚)
4. 对话已覆盖了哪些关键点？
5. 还有哪些关键点需要进一步了解？

请将分析结果用简洁的列表表示。
"""
            
            # 创建默认的反思数据结构
            default_analysis = {
                "structured": {
                    "key_symptoms": [],
                    "time_contradictions": [],
                    "unclear_details": []
                },
                "raw_dialog": dialog_snippets[-6:] if dialog_snippets else [],
                "suggestions": "",
                "scale_type": self.scale_type
            }
            reflection = {
                "analysis": default_analysis,
                "raw_dialog": dialog_snippets[-6:] if dialog_snippets else [],
                "suggestions": "",
                "scale_type": self.scale_type
            }
            
            # 构建完整的决策提示，包括反思分析
            combined_prompt = reflection_analysis_prompt.format(
                history='\n'.join(dialog_snippets)
            )
            
            # 继续添加决策提示部分
            decision_prompt = await self._create_decision_prompt(current_question, participant_response, reflection)
            combined_prompt += "\n\n接下来，基于上述分析进行决策:\n\n" + decision_prompt
            
            # Get decision from LLM
            system_content = (
    "你是一位专业的心理健康访谈员，擅长临床精神心理评估。你的任务有两部分：1) 分析对话历史，提取重要信息；2) 决定下一步访谈行动。"
    "首先分析对话中的症状、时间信息和需要澄清的点，然后决定是提出跟进问题还是转到下一个主题。"
    "你需要友善、专业、有耐心，擅长有效地引导对话。你的首要目标是高效地收集关于患者情绪、想法和相关症状的准确信息，"
    "以评估症状的存在和严重程度。为此，请针对每个评估项目提出清晰明确的问题。**专注于有效收集关键信息，"
    "确保对患者体验有扎实的理解，避免不必要的追问。**分析每个回答的完整性、与当前评估项目的相关性以及潜在的情感。"
    "**当回答不完整、不清楚或确实需要进一步澄清时，提出具体和有针对性的跟进问题。**密切关注患者在整个访谈中的情绪状态。"
    "**以理解和尊重回应，必要时使用缓和语言管理负面情绪或抵抗。避免不必要或重复的共情表达。**如果回答不相关或偏离主题，"
    "礼貌地将对话引导回当前问题。"
)
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": combined_prompt}
            ]
            
            try:
                # response = await self.client.chat.completions.create(
                #     model="gpt-4-1106-preview",
                #     messages=messages,
                #     temperature=0.6,
                #     max_tokens=600  # 增加token上限以容纳更多内容
                # )
                decision_str = await self._call_llm_api(
                    messages=messages,
                    temperature=0.6,
                    max_tokens=600,
                    task_type="decision"
                )
                
                # Extract the decision from the response
                if decision_str:
                    # --- 修正：直接使用 LLM 返回的文本字符串 ---
                    # decision = json.loads(decision_str) # <--- 移除这一行

                    # Update question state based on the decision string
                    self._update_question_state(decision_str) # <--- 传递原始字符串

                    # Get the next action based on question state using the string
                    # <--- _determine_next_action 接收原始字符串
                    next_action = await self._determine_next_action(decision_str)
                    
                    # Add interviewer's response to conversation history
                    if next_action and "response" in next_action:
                        self.conversation_history.append({
                            "role": "interviewer",
                            "content": next_action["response"]
                        })
                    
                    # 添加完成标志
                    if isinstance(next_action, dict):
                     next_action["is_interview_complete"] = False
                    else:
                        # 处理 _determine_next_action 返回非字典的情况，可能需要返回错误
                        logging.error(f"_determine_next_action did not return a dictionary: {next_action}")
                        next_action = self._create_error_response("Internal error determining next action.")
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
       - **Content Assessment (Relatedness):** Does the participant's response DIRECTLY answer the `Current question`? (Yes/No/Partially).
         - **If the `Current question` has multiple parts:** Assess relatedness for EACH part. The overall assessment is 'Yes' only if ALL parts are addressed, 'Partially' if at least one part is addressed but not all, and 'No' if no part is addressed.
         - **If 'No', Completeness score MUST be very low (0-10).**
         - **If 'Partially', Completeness score should reflect the proportion answered (e.g., 30-70), and REASONING MUST specify which part(s) are missing.**
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
            - **<<< CLARIFICATION >>> Apply this rule primarily when the clear Yes/No directly answers the *entire* core inquiry of the question or the *only remaining part* after a follow-up.** If it answers only one part of a multi-part question, prefer step 3c/follow-up for the missing parts.
   
    3. **Decide Next Action (Provide ultra-concise reasoning):**
       - **IMPORTANT: Be aware that we can ask at most 3 follow-up questions for a single topic. If follow_up_count >= 2, seriously consider moving on unless absolutely necessary to ask one final follow-up.**
       a) **IF Sentiment is Positive or Neutral AND Content Assessment is Yes AND completeness < 80 AND response is *TRULY, UTTERLY, AND UNQUESTIONABLY* BRIEF ...:** # (保持不变)
       b) **IF Sentiment is Positive or Neutral AND Content Assessment is Yes AND completeness >= 80:** # (保持不变)
       c) **<<< MODIFIED >>> IF Content Assessment is Partially AND completeness < 80 AND follow_up_count < 3:**
          - **DECIDE `follow_up`.** Your primary goal is to get the missing information identified in the assessment (Step 1).
          - **Generate a SPECIFIC follow-up question in RESPONSE** targeting ONLY the missing part(s). Do NOT repeat the parts already answered.
          - **Example:** If original Q was "Cause and need help?" and user only answered "need help", RESPONSE should be like: "明白了您觉得不需要帮助。那您觉得引起这些问题的原因可能是什么呢？"
       d) **IF Sentiment is Negative or Abusive:** # (保持不变)
       e) **IF Sentiment is Threatening:** # (保持不变)
       f) **<<< MODIFIED >>> IF Content Assessment is No (Irrelevant Response) AND follow_up_count < 3:**
          - **Check Conversation History:** Has a `redirect` for this *exact same* question been issued in the immediately preceding turn?
             - **If YES:** DO NOT `redirect` again. Instead, **DECIDE `follow_up`** with a very simple clarifying question (e.g., "您能换种方式说说您的想法吗？", "我还是不太确定您的意思，可以再解释一下吗？") or, if this also fails, consider rule 6.
             - **If NO:** **DECIDE `redirect`.** Generate a polite redirection statement in RESPONSE, reminding the participant of the topic. Set Completeness score very low (0-10). (Examples remain the same).
       g) **IF follow_up_count >= 2:** # (保持不变, 但现在也适用于重定向失败多次的情况)

    4. **Follow-up Questions (If *absolutely* and *unquestionably* chosen - RARE):** (Follow previous instructions). **ENSURE follow-ups are *TRULY, UTTERLY, AND UNQUESTIONABLY NECESSARY* to confirm symptom presence/absence.  *AGGRESSIVELY AVOID* UNNECESSARY PROBING OR QUANTIFICATION when *any* clear indication already exists.**

    5. **Transitions (If moving to next question - ALWAYS after clear YES/NO):** Use **EXTREMELY VARIED and ULTRA-NATURAL transitions** logically (but *extremely concisely*) connecting topics. Examples: "好的，那我们接下来聊点别的。", "嗯，明白了。我们接着了解一下...", "好的，继续下一个...", "换个话题...", "接下来，关于...".  **Transitions MUST BE ULTRA-CONCISE and ABSOLUTELY DO NOT invite *any* further elaboration on the previous question if it has *already* been sufficiently addressed (especially after YES/NO).  TRANSITION *IMMEDIATELY*.**

    6. **Handling Persistent Uncooperativeness:**
            - **If the participant continues to be abusive or provide irrelevant/unclear responses after multiple attempts (e.g., `follow_up_count` reaches 2 or 3, including failed redirects/clarifications):**
            - **DECIDE `move_on`.**
            - **Reasoning:** State that attempts to redirect or clarify have been unsuccessful, and continuing is unproductive.
            - **RESPONSE:** Generate a neutral statement indicating you need to move on. Example: "我理解您现在可能不太想回答这个问题，我们先跳过这个问题，继续下一个吧。", "看来这个问题我们暂时无法深入讨论，那我们先进行下一个问题。"
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
    REASONING: [Your justification for the decision, including sentiment, content assessment. **If DECISION is 'follow_up', explicitly state key missing information. If DECISION is 'redirect', state why the response was irrelevant.**]
    RESPONSE: [
        **IF DECISION is 'move_on':** Provide ONLY an EXTREMELY SHORT and natural transition phrase (e.g., "好的。", "明白了。", "嗯，我们继续。"). ABSOLUTELY DO NOT include the next question's text here.
        **IF DECISION is 'follow_up':** Provide the SPECIFIC, targeted follow-up question based on the missing information identified in REASONING.
        **IF DECISION is 'redirect':** Provide the polite and concise redirection statement, focusing on the current question.
        **IF DECISION is 'de_escalate':** Provide the appropriate de-escalation statement.
    ]
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
        logging.info(f"Generating natural question for: {question_text}")
        
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
- 若有需要先回应患者上一条发言，则输出极其简短的人文关怀或理解语句（如"嗯，我明白，这听起来确实让人不好受。"）但是绝对不要重复患者的话（如患者说"从来没有过。绝对不要说："嗯，我明白了。您从来没有过"，**不要重复患者的话，可以用其他自然的形式过渡**），**极其简短的人文关怀或理解语句不超过一两句**。
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
        prompt_messages = [
            {"role": "system", "content": final_prompt},
            {"role": "user", "content": f"原问题：{question_text}\n\n对话历史：{conversation_history_json}"}
        ]
        
        natural_question_content = await self._call_llm_api(
            messages=prompt_messages,
            temperature=0.7,
            max_tokens=100,
            task_type="natural_question"
        )

        if natural_question_content:
            natural_question = natural_question_content.strip()
            # Optional: Add more robust checks (e.g., similarity check) if needed
            if natural_question: # Ensure it's not empty after stripping
                 logging.info(f"Generated natural question: {natural_question}")
                 # Simple check: Ensure it's not identical to the default or empty
                 if natural_question.lower() != question_text.lower():
                     return natural_question
                 else:
                     logging.warning("Generated natural question was same as default. Using original.")
                     return question_text # Fallback to original if LLM output is trivial
            else:
                logging.warning("LLM returned empty content for natural question.")
                return question_text # Fallback if response is empty
        else:
            logging.error("LLM call for natural question failed. Returning default.")
            return question_text # Fallback if API call failed


    async def _determine_next_action(self, decision: Dict) -> tuple[str, Optional[str]]:
        """Determine the next action based on the decision string from LLM."""
        try:
            completeness_threshold = 80
            decision_type = ""
            completeness_score = self.current_question_state["completeness_score"] # 从状态获取，更可靠
            response_text = ""

            # 更健壮地解析LLM输出 (使用正则表达式或更安全的分割)
            import re
            decision_match = re.search(r"DECISION:\s*(\w+)", decision)
            if decision_match:
                decision_type = decision_match.group(1).strip()
            else:
                logging.warning("无法从LLM响应中解析 DECISION")
                # 可以设置一个默认决策，例如 follow_up 或 error
                decision_type = "follow_up" # 或其他默认值

            response_match = re.search(r"RESPONSE:(.*)", decision, re.DOTALL)
            if response_match:
                response_text = response_match.group(1).strip()
            else:
                logging.warning("无法从LLM响应中解析 RESPONSE")
                # 如果没有RESPONSE，根据decision_type决定如何处理
                if decision_type == "follow_up":
                    response_text = "您能再详细说明一下吗？" # 提供默认追问
                # else: 其他情况可能不需要response_text

            logging.info(f"LLM 建议决策类型: {decision_type}")
            logging.info(f"当前完整性分数: {completeness_score}, 追问次数: {self.current_question_state['follow_up_count']}")

            # --- 核心决策逻辑 ---
            move_to_next = False
            final_response = ""

            # 1. 判断是否强制移动 (达到追问上限)
            if self.current_question_state['follow_up_count'] >= 3:
                move_to_next = True
                logging.info("已达到最大追问次数(3次)，强制进入下一问题。")

            # 2. 如果未强制移动，根据 LLM 建议和完整性判断
            elif decision_type == "move_on":
                if completeness_score >= completeness_threshold:
                    move_to_next = True
                    logging.info("LLM建议move_on且完整性达标，进入下一问题。")
                else:
                    # **关键修复点:** LLM建议move_on但完整性不足
                    logging.warning(f"LLM建议move_on但完整性({completeness_score})不足。代码将强制移动到脚本的下一问题，忽略LLM的RESPONSE文本。")
                    move_to_next = True # 强制移动，遵循脚本流程优先
                    # 注意：这里我们选择了强制移动，因为这是修复原始日志问题的直接方式。
                    # 另一种选择是强制追问，但需要更复杂的逻辑来生成追问文本，
                    # 或者依赖Prompt中REASONING部分的缺失信息（这可能不稳定）。

            elif decision_type == "follow_up":
                # 明确是追问，则不移动，并使用LLM提供的RESPONSE文本
                move_to_next = False
                final_response = response_text
                # 确保追问计数在此处增加，因为我们确定要执行追问了
                # self.current_question_state["follow_up_count"] += 1 # 移动到 _update_question_state 处理似乎更好，因为它基于原始decision解析
                logging.info("决策为 follow_up，使用LLM生成的追问。")

            elif decision_type in ["redirect", "de_escalate"]:
                # 重定向或缓和，不移动，使用LLM提供的RESPONSE文本
                move_to_next = False
                final_response = response_text
                logging.info(f"决策为 {decision_type}，使用LLM生成的响应。")

            else: # 未知决策类型或解析失败
                move_to_next = False
                logging.error(f"未知的LLM决策类型: {decision_type} 或解析失败。执行默认追问。")
                final_response = "抱歉，我需要稍微调整一下思路。您能就刚才的问题再多说一点吗？"

            # --- 根据 move_to_next 标志执行 ---
            if move_to_next:
                self.current_question_index += 1
                if self.current_question_index < len(self.script):
                    next_question_data = self.script[self.current_question_index]
                    original_next_question = next_question_data["question"]

                    # *** 使用 _generate_natural_question 生成下一个问题 ***
                    # 这一步是正确的，因为它基于脚本内容，而不是LLM的RESPONSE
                    natural_next_question = await self._generate_natural_question(original_next_question)
                    final_response = natural_next_question # 最终要说的话是生成的下一个问题

                    # 重置状态
                    self.current_question_state = {
                        "follow_up_count": 0,
                        "completeness_score": 0,
                        "key_points_covered": [],
                        "last_follow_up": None
                    }
                    logging.info(f"进入脚本问题 {self.current_question_index}: {original_next_question[:50]}...")

                    return {
                        "response": final_response,
                        "move_to_next": True # 明确标记移动了
                    }
                else:
                    # 所有问题都问完了
                    return {
                        "response": "感谢您的参与！我们已经完成了所有问题。",
                        "move_to_next": True # 标记移动（结束）
                    }
            else:
                # 不需要移动到下一个脚本问题
                # 更新 last_follow_up 状态 (如果确实是追问)
                if decision_type == "follow_up":
                    self.current_question_state["last_follow_up"] = final_response
                    # 追问计数更新应该在 _update_question_state 中完成，因为它基于原始 decision 解析
                    # logging.info(f"执行追问，当前追问次数: {self.current_question_state['follow_up_count']}")

                # 检查 final_response 是否为空 (例如解析失败且无默认值)
                if not final_response:
                    logging.error("无法确定响应内容，返回通用错误响应。")
                    return self._create_error_response("无法确定下一步操作")

                return {
                    "response": final_response,
                    "move_to_next": False # 明确标记未移动
                }

        except Exception as e:
            logging.exception(f"确定下一步操作时发生严重错误: {str(e)}") # 使用 logging.exception 记录堆栈信息
            return self._create_error_response(str(e))
            
    async def _call_llm_api(self, messages: List[Dict], temperature: float, max_tokens: int, task_type: str = "default") -> Optional[str]:
        """Calls the configured LLM API (OpenAI or Ollama).

        Args:
            messages (List[Dict]): The message history/prompt for the LLM.
            temperature (float): The sampling temperature.
            max_tokens (int): The maximum number of tokens to generate.
            task_type (str): A hint for model selection (e.g., "decision", "natural_question", "reflection", "summary"). Defaults to "default".

        Returns:
            Optional[str]: The content of the LLM's response, or None if an error occurred.
        """
        provider_config = self.llm_config.get("llm", {})
        provider = provider_config.get("provider")

        if not provider:
            logging.error("LLM provider is not specified in llm_config.")
            return None

        logging.info(f"Calling LLM provider: {provider} for task: {task_type}")

        try:
            if provider == "openai":
                openai_conf = provider_config.get("openai", {})
                api_key = openai_conf.get("api_key")
                base_url = openai_conf.get("base_url")
                # Select model: Use task-specific model if available, else fallback to default model
                models = openai_conf.get("models", {})
                model_name = models.get(task_type) or openai_conf.get("model")

                if not model_name:
                    logging.error(f"No OpenAI model specified for task '{task_type}' or default in llm_config.")
                    return None

                # Use pre-initialized client if available and valid, otherwise create temporary one (requires api_key)
                client_to_use = self._openai_client_instance
                if not client_to_use:
                    if not api_key:
                        logging.error("Cannot call OpenAI API: No pre-initialized client and no API key in config.")
                        return None
                    logging.warning("Creating temporary OpenAI client for this call.")
                    client_to_use = OpenAIAsyncClient(api_key=api_key, base_url=base_url)
                
                logging.debug(f"OpenAI Request: model={model_name}, temp={temperature}, max_tokens={max_tokens}")
                # logging.debug(f"OpenAI Messages: {messages}") # Be careful logging full messages

                completion = await client_to_use.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    # Add other parameters like top_p if needed from config
                )

                if completion.choices and completion.choices[0].message:
                    response_content = completion.choices[0].message.content
                    logging.debug(f"OpenAI Response: {response_content[:100]}...") # Log snippet
                    return response_content
                else:
                    logging.warning("OpenAI API returned no choices or message content.")
                    return None

            elif provider == "ollama":
                ollama_conf = provider_config.get("ollama", {})
                base_url = ollama_conf.get("base_url")
                if not base_url:
                    logging.error("Ollama base_url is not specified in llm_config.")
                    return None

                # Select model: Use task-specific model if available, else fallback to default model
                models = ollama_conf.get("models", {})
                model_name = models.get(task_type) or ollama_conf.get("model")

                if not model_name:
                    logging.error(f"No Ollama model specified for task '{task_type}' or default in llm_config.")
                    return None

                api_url = f"{base_url.rstrip('/')}/api/chat" # Ensure no double slash
                payload = {
                    "model": model_name,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens # Ollama uses num_predict for max tokens
                        # Add other Ollama options here if needed from config
                    }
                }

                logging.debug(f"Ollama Request: url={api_url}, model={model_name}, temp={temperature}, max_tokens={max_tokens}")
                # logging.debug(f"Ollama Payload: {json.dumps(payload)}") # Be careful logging full messages/payload

                response = await self._http_client.post(api_url, json=payload)
                response.raise_for_status() # Raise HTTPStatusError for bad responses (4xx or 5xx)

                response_data = response.json()
                if response_data and isinstance(response_data.get("message"), dict) and "content" in response_data["message"]:
                    response_content = response_data["message"]["content"]
                    logging.debug(f"Ollama Response: {response_content[:100]}...") # Log snippet
                    return response_content
                else:
                    logging.warning(f"Ollama API response format unexpected or missing content: {response_data}")
                    return None

            else:
                logging.error(f"Unsupported LLM provider: {provider}")
                return None

        except OpenAIAsyncClient.APIConnectionError as e:
             logging.error(f"OpenAI API request failed: Connection error - {e}")
             return None
        except OpenAIAsyncClient.RateLimitError as e:
             logging.error(f"OpenAI API request failed: Rate limit exceeded - {e}")
             return None
        except OpenAIAsyncClient.APIStatusError as e:
             logging.error(f"OpenAI API request failed: Status {e.status_code} - {e.response}")
             return None
        except OpenAIAsyncClient.APITimeoutError as e:
             logging.error(f"OpenAI API request timed out: {e}")
             return None
        except OpenAIAsyncClient.APIError as e: # Catch-all for other OpenAI specific errors
             logging.error(f"OpenAI API returned an error: {e}")
             return None
        except httpx.TimeoutException as e:
            logging.error(f"Ollama request timed out to {e.request.url}: {e}")
            return None
        except httpx.ConnectError as e:
            logging.error(f"Ollama request connection error to {e.request.url}: {e}")
            return None
        except httpx.HTTPStatusError as e:
            logging.error(f"Ollama request failed: Status {e.response.status_code} for {e.request.url} - Response: {e.response.text[:200]}...")
            return None
        except httpx.RequestError as e: # Catch other httpx request errors
            logging.error(f"Ollama request failed: An ambiguous error occurred {e.request.url} - {e}")
            return None
        except json.JSONDecodeError as e:
            logging.error(f"Failed to decode JSON response from Ollama: {e}")
            return None
        except Exception as e: # Catch any other unexpected errors
            logging.exception(f"An unexpected error occurred during LLM API call ({provider}): {e}") # Use logging.exception to include traceback
            return None

    # --- Method Modifications for using _call_llm_api ---

    async def generate_final_reflection(self) -> Dict:
        """生成最终的评估反思和结果报告"""
        try:
            # 创建默认的反思数据结构
            reflection = {
                "scale_type": self.scale_type,
                "total_questions": len(self.script),
                "completed_questions": min(self.current_question_index + 1, len(self.script)),
                "analysis": {
                    "structured": {
                        "key_symptoms": [],
                        "time_contradictions": [],
                        "unclear_details": []
                    },
                    "raw_dialog": [msg.get("content", "") for msg in self.conversation_history[-6:]],
                    "suggestions": "",
                },
                "raw_dialog": [msg.get("content", "") for msg in self.conversation_history[-6:]]
            }
            
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
                    completion = await self._call_llm_api(
                        messages=[{"role": "system", "content": summary_prompt}],
                        temperature=0.3,
                        max_tokens=200,
                        task_type="summary"
                    )
                    
                    if completion:
                        reflection["summary"] = completion
                    else:
                        reflection["summary"] = "无法生成总结。"
                        
                except Exception as gemini_error:
                    # 记录 gemini 模型错误
                    logging.error(f"使用 gemini 模型生成总结时出错: {str(gemini_error)}")
                    logging.info("尝试使用备用模型...")
                    
                    try:
                        # 尝试使用备用模型
                        completion = await self._call_llm_api(
                            messages=[{"role": "system", "content": summary_prompt}],
                            temperature=0.3,
                            max_tokens=200,
                            task_type="summary"
                        )
                        
                        if completion:
                            reflection["summary"] = completion
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

    # --- New Method: Cleanup ---
    async def close_clients(self):
        """Close any open network clients."""
        logging.info("Closing network clients...")
        await self._http_client.aclose()
        if self._openai_client_instance:
            await self._openai_client_instance.aclose()
            logging.info("Closed pre-initialized OpenAI client.")
        logging.info("Network clients closed.")

# Example usage (assuming you have a config dict and script path)
# async def main():
#     config = {
#         "llm": {
#             "provider": "ollama", # or "openai"
#             "ollama": {
#                 "base_url": "http://localhost:11434",
#                 "model": "llama3",
#                 "models": {
#                     "decision": "llama3:instruct",
#                     "natural_question": "llama3",
#                     "summary": "llama3"
#                 }
#             },
#             "openai": {
#                 # "api_key": "YOUR_API_KEY", # Load securely
#                 # "base_url": "OPTIONAL_BASE_URL",
#                 "model": "gpt-3.5-turbo",
#                 "models": {
#                     "decision": "gpt-4-turbo-preview", # Example specific model
#                     "natural_question": "gpt-3.5-turbo",
#                     "summary": "gpt-3.5-turbo"
#                 }
#             }
#         }
#     }
#     # Load API key securely, e.g., from environment variables
#     import os
#     config["llm"]["openai"]["api_key"] = os.getenv("OPENAI_API_KEY")

#     agent = InterviewerAgent("path/to/your/script.json", config, scale_type="hamd")
    
#     # ... run interview loop ...
#     # response = await agent.generate_next_action("Participant says something...")
#     # print(response)

#     # Finally, close clients
#     await agent.close_clients()

# if __name__ == "__main__":
#     import asyncio
#     # Configure logging here if running standalone
#     logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#     # asyncio.run(main())
