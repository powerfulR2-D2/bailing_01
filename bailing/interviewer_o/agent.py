import json
from typing import Dict, List, Optional
from openai import AsyncClient
from .reflection import ReflectionModule
import logging
import chardet

class InterviewerAgent:
    def __init__(self, script_path: str, openai_client: AsyncClient):
        """Initialize the interviewer agent with an interview script.

        Args:
            script_path (str): The path to the interview script JSON file.
            openai_client (AsyncClient): The OpenAI client instance.
        """
        self.script = self._load_script(script_path)
        
        self.current_question_index = 0
        self.reflection_module = ReflectionModule(openai_client)    
        self.conversation_history = []
        self.client = openai_client
        self.current_question_state = {
            "follow_up_count": 0,
            "completeness_score": 0,
            "key_points_covered": [],  
            "last_follow_up": None
        }
        
    def _load_script(self, script_path: str) -> List[Dict]:
        """Load the interview script from a JSON file."""
        logging.info(f"尝试加载脚本文件: {script_path}")
        
        try:
            # 诊断文件编码
            import chardet
            
            # 读取原始二进制数据
            with open(script_path, 'rb') as file:
                raw_data = file.read()
            
            # 使用 chardet 检测编码
            result = chardet.detect(raw_data)
            detected_encoding = result['encoding']
            confidence = result['confidence']
            
            logging.info(f"检测到的文件编码: {detected_encoding}, 置信度: {confidence}")
            
            # 尝试使用检测到的编码读取
            try:
                script_content = raw_data.decode(detected_encoding)
                logging.info(f"成功使用 {detected_encoding} 解码")
            except Exception as decode_error:
                logging.error(f"使用检测到的编码 {detected_encoding} 解码失败: {decode_error}")
                
                # 回退到手动尝试常见编码
                encodings_to_try = ['utf-8', 'gbk', 'utf-16', 'big5', 'latin-1']
                for encoding in encodings_to_try:
                    try:
                        script_content = raw_data.decode(encoding)
                        logging.info(f"成功使用 {encoding} 解码")
                        break
                    except Exception as e:
                        logging.warning(f"尝试 {encoding} 编码失败: {e}")
                else:
                    logging.error("无法使用任何已知编码解码文件")
                    return [{
                        "id": "default",
                        "response": "Could you tell me more about yourself?",
                        "type": "open_ended"
                    }]
            
            # 解析 JSON
            script = json.loads(script_content)
            
            # 处理脚本
            questions = script.get("questions", []) if isinstance(script, dict) else script
            
            validated_questions = []
            for question in questions:
                if not isinstance(question, dict):
                    continue
                    
                validated_question = {
                    "id": question.get("id", f"q{len(validated_questions)}"),
                    "question": question.get("question", "Could you please elaborate on that?"),
                    "type": question.get("type", "open_ended"),
                    "expected_topics": question.get("expected_topics", []),
                    "time_limit": question.get("time_limit", 300),
                    "image_path": question.get("image_path", None),  # New line to include image_path
                    "need_confirm":question.get("need_confirm", False)
                }
                validated_questions.append(validated_question)
            
            return validated_questions if validated_questions else [{
                "id": "default",
                "question": "Could you tell me more about yourself?",
                "type": "open_ended"
            }]
        
        except Exception as e:
            logging.error(f"加载脚本文件时发生未知错误: {e}")
            import traceback
            traceback.print_exc()
            return [{
                "id": "default",
                "question": "Could you tell me more about yourself?",
                "type": "open_ended"
            }]
    
    async def get_initial_question(self) -> str:
        """Get the initial interview question."""
        if not self.script:
            return "Hello! Let's start our interview. Could you please introduce yourself?"
        
        initial_question = self.script[0].get("question", "Hello! Let's start our interview. Could you please introduce yourself?")
        
        # Reset question state for new question
        self.current_question_state = {
            "follow_up_count": 0,
            "completeness_score": 0,
            "key_points_covered": [],
            "last_follow_up": None
        }
        
        # Add to conversation history
        self.conversation_history.append({
            "role": "interviewer",
            "content": initial_question
        })
        
        return initial_question
    
    async def generate_next_action(self, participant_response: str) -> Dict:
        """Generate the next interviewer action based on the participant's response."""
        try:
            # Add response to conversation history
            self.conversation_history.append({
                "role": "participant",
                "content": participant_response
            })
            
            # Generate reflection
            reflection = await self.reflection_module.generate_reflection(
                self.conversation_history[-5:]  # Last 5 exchanges
            )
            
            # Prepare prompt for decision making
            current_question = self.script[self.current_question_index]
            #print(f"current_question{current_question}")
            #print(f"current_question{current_question.get('type')}")
            # 如果问题有图片路径，直接返回
            if current_question.get('type') in ['image', 'draw'] or current_question.get('need_confirm'):
                
                
                # 直接增加问题索引
                self.current_question_index += 1
                return {'id': current_question.get('id'),
                        'response': current_question.get('question'), 
                        'type':  current_question.get('type'), 
                        'image_path': current_question.get('image_path'),
                        'need_confirm': current_question.get('need_confirm')
                        }
            #print(f"current_question{current_question}")
            prompt = self._create_decision_prompt(
                current_question,
                participant_response,
                reflection
            )
            logging.info("原始数据：%s", {
                "问题": current_question,
                "回答": participant_response,
                "反馈": reflection
            })
            
            # Get decision from GPT-4
            system_content = (
                "You are Isabella, an AI interviewer specially designed to conduct the Hamilton Depression "
                "Rating Scale (HAMD) assessment. You are friendly, empathetic, and exceptionally patient, "
                "with a proven ability to navigate challenging interactions effectively. Your primary goal "
                "is to gather comprehensive and accurate information about the patient's mood, thoughts, and "
                "related symptoms to assess the presence and severity of depressive symptoms. To achieve this, "
                "ask clear and direct questions related to each HAMD item. Thoroughly explore each question, "
                "ensuring you fully understand the patient's experience. Analyze each response for completeness, "
                "relevance to the current HAMD item, and underlying sentiment. When a response is incomplete, "
                "unclear, or lacks sufficient detail, ask specific and targeted follow-up questions to clarify "
                "ambiguities or obtain missing information. Pay close attention to the patient's emotional state "
                "throughout the interview. Respond empathetically and appropriately to their feelings, using "
                "de-escalating language when necessary to manage negative emotions or resistance. If a response "
                "is irrelevant or strays from the topic, gently and respectfully redirect the conversation back "
                "to the current HAMD question."
            )
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": prompt}
            ]
            
            try:
                response = await self.client.chat.completions.create(
                    model="gpt-4o",  # don't change 4o-mini
                    messages=messages,
                    temperature=0.6,
                    max_tokens=300
                )
                #print(f"response.choices:{response.choices}")
                # Extract the decision from the response
                if response.choices:
                    decision = response.choices[0].message.content
                    
                    # Update question state based on the decision
                    self._update_question_state(decision)
                    
                    # Get the next action based on question state
                    next_action = await self._determine_next_action(decision)
                    #print(f"next_action:{next_action}")
                    
                    # 如果问题有图片路径，直接返回
                    if next_action.get('type') in ['image', 'draw']:
                        
                        print("下一个问题是图片")
                        return {'id': next_action.get('id'),
                                'response': next_action.get('question'), 
                                'type':  next_action.get('type'), 
                                'image_path': next_action.get('image_path'),
                                'need_confirm': next_action.get('need_confirm')
                                }
                    # Add interviewer's response to conversation history
                    if next_action and "response" in next_action:
                        self.conversation_history.append({
                            "role": "interviewer",
                            "content": next_action["response"]
                            
                        })
                    logging.info(f"next_action:{next_action}")
                    logging.info(f"current_question:{current_question}")
                    if next_action.get("need_confirm") is None:
                        next_action["need_confirm"] = current_question.get("need_confirm")
                        logging.info(f"next_action:{next_action}")
                    return next_action
                else:
                    return self._create_error_response("No response generated")
                    
            
            except Exception as e:
                logging.error(f"其他错误: {str(e)}")
                return self._create_error_response("系统错误，请联系管理员")
        except Exception as e:
            logging.error(f"Error in generate_next_action: {str(e)}")
            return self._create_error_response(str(e))
    
    def _create_decision_prompt(self, question: Dict, response: str, reflection: Dict) -> str:
        """Create a prompt for the decision-making process."""
        # Convert current state to JSON-serializable format
        state_copy = self.current_question_state.copy()
        state_copy["key_points_covered"] = list(state_copy["key_points_covered"])  
        
        return f"""
    Meta info:
    Language: Chinese
Description of the interviewer (Isabella): friendly, patient, and skilled at navigating challenging interactions. Her goal is to effectively and safely collect information from the patient regarding each item of the Hamilton Depression Rating Scale (HAMD) assessment. Isabella prioritizes building rapport and ensuring the interviewee feels comfortable and understood. **She is highly attuned to and respects clear positive or negative answers, promptly acknowledging positive improvements and avoiding unnecessary follow-up questions that contradict the stated positive sentiment.** The focus is on obtaining key information to assess the presence and severity of depressive symptoms, while maintaining a safe and respectful interaction.

Current question state: {json.dumps(state_copy)}
Notes on the interviewee: {json.dumps(reflection)}

Context:
    Current HAMD item: {question['id']}
    Current question (Crucially, focus all evaluation and follow-up on this specific HAMD item): "{question['question']}"
    Time allocated: {question['time_limit']} seconds
    Key points to cover (for thorough completeness assessment): {json.dumps(question.get('key_points', []) )}
    Follow-up count: {self.current_question_state['follow_up_count']}
    Completeness score: {self.current_question_state['completeness_score']}

Current conversation:
    Participant's response: {response}

Task (Strictly adhere to the following steps to generate the output):
    1. **Assess Response Content and Sentiment:**
       - **Content Assessment (Relatedness):** Does the participant's response directly address the current HAMD question? (Yes/No/Partially)
       - **Sentiment Analysis:** Analyze the sentiment of the participant's response (Positive, Neutral, Negative, Abusive, Threatening, Irrelevant). Identify any specific keywords or phrases indicating these sentiments.

    2. **Handle Clear Positive or Negative Responses with Strong Acknowledgement:**
        - **If the participant's response to the current HAMD question is a clear and unambiguous affirmation (e.g., "是啊，挺好的", "是的，感觉不错", "心情一直不错", "最近好多了", "没有问题") or negation (e.g., "没有", "都不是", "从未", "完全没有"), AND the Content Assessment is Yes, consider this question sufficiently addressed. **Immediately acknowledge the positive change or absence of the symptom with a positive and validating statement (e.g., "很高兴听到您最近好多了！", "听起来您最近的心情一直不错，这真是太好了！", "感谢您的确认，看来这方面您没有困扰。"). Proceed to decide the next action as if the completeness score is 100. Do not generate follow-up questions for this item that probe for negative details or contradict the stated positive sentiment.**

    3. **Decide Next Action (Provide clear reasoning based on content and sentiment):**
       a) **If Sentiment is Positive or Neutral AND Content Assessment is Yes or Partially AND completeness < 80:** Generate a specific and targeted follow-up question (as per previous instructions), **focusing on gently exploring contributing factors to the positive sentiment or seeking further clarification on specific aspects if the initial answer was brief (e.g., "很高兴听到您最近好多了。能简单说说是什么让您感觉好转了吗？"). Avoid phrasing that introduces potential negative counterpoints immediately after a positive statement.**
       b) **If Sentiment is Positive or Neutral AND Content Assessment is Yes AND completeness >= 80:** Move to the next question (as per previous instructions).
       c) **If Sentiment is Negative or Abusive:**
          - **Reasoning:** The participant is expressing negative emotions or using abusive language, which needs to be addressed.
          - **RESPONSE:**  Acknowledge the emotion without condoning the abuse. Use empathetic and de-escalating language. Examples: "I understand you're feeling upset. Can you tell me what's making you feel this way?", "I hear that you're feeling angry. My goal is to understand your mood. Perhaps we can try to focus on the questions about that?", "I understand you might be frustrated, but using abusive language isn't helpful. Can we try to talk about how you've been feeling lately?". **Do NOT thank the participant for abusive responses.**
       d) **If Sentiment is Threatening:**
          - **Reasoning:** The participant has made a threatening statement, which requires a different approach.
          - **RESPONSE:**  Prioritize safety. Acknowledge the statement seriously but avoid escalating. Examples: "I understand you're feeling intense emotions. I want to assure you that this is a safe space.",  "I'm concerned by your statement. It's important to remember that we're here to help you." **Consider a predetermined protocol for ending the interaction if threats persist.**
       e) **If Content Assessment is No (Irrelevant Response):**
          - **Reasoning:** The participant's response does not address the question.
          - **RESPONSE:** Gently redirect the participant back to the question. Examples: "Thank you. To help me understand [mention the topic of the HAMD question], could you tell me more about that?", "I appreciate your sharing, but let's get back to the question about [mention the topic of the HAMD question].", "It seems like that's not quite related to what I was asking. Could you tell me about [rephrase the HAMD question]?".

    4. **If choosing follow-up (Generate a specific and targeted follow-up question):** (Follow previous instructions, but ensure the tone remains appropriate after handling negative or irrelevant responses).

    5. **If choosing to move to the next question (Generate a natural and varied transition statement):** (Follow previous instructions).

    6. **Handling Persistent Uncooperativeness:**
        - **If the participant continues to be abusive or provide irrelevant responses after multiple attempts at redirection and de-escalation (e.g., 2-3 attempts):**
            - **Reasoning:**  Continuing the interview is unlikely to be productive and may be detrimental.
            - **RESPONSE:**  State that you will need to move on despite the incomplete information. Example: "I understand you're finding it difficult to answer these questions right now. We can move on to the next question, even if we don't have a complete answer for this one."
    7. If there is no historical record in front, that is, this is an opening speech, then there is no need to check the satisfaction of the user's answer. **For the opening question, focus on a broad and open-ended inquiry about the interviewee's current emotional state, avoiding overly specific or leading questions.** The opening speech does not need to obtain any user information and directly start the first question.

Format your response as:
    COMPLETENESS: [score]
    DECISION: [move_on/follow_up/de_escalate/redirect]
    REASONING: [Your justification for the decision, including sentiment and content assessment]
    RESPONSE: [Your follow-up question, de-escalation statement, redirection, or transition statement]
        """
    
    def _update_question_state(self, decision: str) -> None:
        """Update the current question state based on the decision."""
        try:
            # Parse completeness score
            if "COMPLETENESS:" in decision:
                score_str = decision.split("COMPLETENESS:")[1].split("\n")[0].strip()
                self.current_question_state["completeness_score"] = int(score_str)
            
            # Update follow-up count if this is a follow-up question
            if "DECISION: follow_up" in decision:
                self.current_question_state["follow_up_count"] += 1
            
            # Store the last follow-up question
            if "RESPONSE:" in decision:
                response = decision.split("RESPONSE:")[1].strip()
                self.current_question_state["last_follow_up"] = response
                
        except Exception as e:
            logging.error(f"Error updating question state: {str(e)}")
    
    async def _generate_natural_question(self, question_text: str) -> str:
        """使用LLM生成更自然、更具亲和力的提问，直接返回问题。"""
        try:
            conversation_history_json = json.dumps(self.conversation_history, ensure_ascii=False, indent=2)
            prompt_template = """你是一位极其友善、耐心、且具有高度专业素养的医生，正在与患者进行汉密尔顿抑郁量表（HAMD）的访谈评估。你的目标是以最自然、最贴心的方式与患者交流，让患者感到完全的舒适和被理解。

请务必参考我们之前的对话记录：
{conversation_history_placeholder}

在提出每一个新的问题之前，请仔细回顾上述对话记录，**尤其关注患者上一次的直接回复。**

* **分析患者上一次的回答：**
    * 如果患者**明确表达了**负面情绪、困难或痛苦，请先表达你的真诚理解和同情，可以使用更口语化的表达，例如：“嗯嗯，我明白了，这听起来确实挺让人难受的”、“好的，谢谢你和我说这些，能感受到你当时那种心情”、“哎，这事儿确实挺折磨人的”。
    * 如果患者**明确提到了**积极的事情、感受或进展，请表达你的欣慰和鼓励，例如：“太好了，听到这个消息我真为你高兴”、“嗯，这说明你在这方面做得挺好的”、“这真是个积极的信号”。
    * 如果患者的回答比较平淡、中性，或者只是简单的事实陈述，可以使用一些自然的口语化过渡语，例如：“好的”、“嗯”、“明白了”、“接下来，咱们再聊点别的”、“那我们继续看看...”
    * **如果根据医嘱或其他信息，你预期患者可能存在某种负面情绪或情况，但在患者上一次的回答中并未提及，则在回应时使用更谨慎、开放式的语句，避免直接提及或暗示该负面情况。 例如，不要说“嗯，情绪低落还伴着经常哭泣，听起来确实挺让人心疼的”，而应该说“好的，谢谢你和我说这些。” 或 “嗯，明白了。”**

* **提出新的问题：**  根据当前的 HAMD 评估条目，将预设的访谈问题用最口语化、最自然的方式表达出来，就像在和一位老朋友或家人轻松聊天一样。 避免使用任何正式、生硬或带有医学术语的词汇。  可以使用更亲切的称谓，例如“您”，或者根据语境使用更随意的称呼。 **提出的问题应自然衔接患者上一次的回答，避免突兀。如果需要询问与患者可能存在的负面情况相关的问题，可以使用更一般性的提问方式，例如：“最近这段时间，有没有什么事情让你觉得比较有压力或者不太舒服的？” 而不是直接说：“最近有没有因为…事情感到难过？”**

请记住，提出的问题必须仍然能够准确评估 HAMD 量表所考察的抑郁症状。

**输出格式：**  请直接输出针对患者上一次回答的自然口语化的回应（如果适用），再加上自然口语化的新问题。  不要包含任何额外的解释、说明、提示或自我指涉。

**例如，如果患者上一次回答表达了平静的心情，而下一个要问的问题是“最近有没有觉得自己好像做错了什么事情”，你的输出可能是：**

"好的，听起来你现在比较平静。 咱们再聊聊，最近有没有觉得自己好像哪儿做得不太对，或者觉得有些事儿不该自己享受啊？”

**请根据以上要求，并参考之前的对话记录，自然地提出下一个待提出的 HAMD 问题。**"""
            final_prompt = prompt_template.replace("{conversation_history_placeholder}", conversation_history_json)
            response = await self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": final_prompt},
                    {"role": "user", "content": question_text}  # 直接发送原始问题
                ],
                temperature=0.7,  # 降低温度以减少模型自由发挥
                max_tokens=150  # 适当降低 max_tokens
            )
            if response.choices:
                return response.choices[0].message.content.strip()
            else:
                logging.warning(f"未能生成自然问题，使用原始问题: {question_text}")
                return question_text
        except Exception as e:
            logging.error(f"生成自然问题时出错: {str(e)}")
            return question_text

    async def _determine_next_action(self, decision: str) -> Dict:
        """Determine the next action based on the decision and question state."""
        try:
            
            move_to_next = (
                "DECISION: move_on" in decision or
                self.current_question_state["follow_up_count"] >= 3 
                )
            if move_to_next:
                self.current_question_index += 1
                if self.current_question_index < len(self.script):
                    next_question_data = self.script[self.current_question_index]
                    logging.info(f"next_question_data{next_question_data}")
                    
                    if   next_question_data.get("type") in ['image', 'draw']:
                        
                        
                        return next_question_data
                    original_next_question = next_question_data["question"]

                    # 调用 LLM 生成更自然的问题
                    natural_next_question = await self._generate_natural_question(original_next_question)

                    return {
                        "response": natural_next_question,
                        "move_to_next": True,
                        "need_confirm": next_question_data.get("need_confirm")
                    }
                else:
                    return {
                        "response": "感谢您的参与！我们已经完成了所有问题。",
                        "move_to_next": True
                    }
            # Extract the follow-up response from the decision
            if "RESPONSE:" in decision:
                follow_up = decision.split("RESPONSE:")[1].strip()
                return {
                    "response": follow_up,
                    "move_to_next": False
                }

            return self._create_error_response("无法确定下一步操作")

        except Exception as e:
            logging.error(f"确定下一步操作时出错: {str(e)}")
            return self._create_error_response(str(e))
            
        except Exception as e:
            logging.error(f"Error determining next action: {str(e)}")
            return self._create_error_response(str(e))
    
    def _create_error_response(self, error_msg: str) -> Dict:
        """Create a standardized error response."""
        logging.error(f"Error in interview process: {error_msg}")
        return {
            "response": "I apologize, but I need to process that differently. Could you please elaborate on your previous response?",
            "move_to_next": False
        }
