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
                    model="gemini-2.0-pro-exp-02-05",
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
        
        # 构建上下文部分
        dialog_context = '\n'.join(reflection_report.get('raw_dialog', [])[-4:])
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
                try:
                    # More robust score extraction with validation
                    score = int(score_str)
                    # Ensure score is within valid range (0-100)
                    if 0 <= score <= 100:
                        self.current_question_state["completeness_score"] = score
                    else:
                        logging.warning(f"Completeness score out of range: {score}, defaulting to previous value")
                except ValueError:
                    logging.warning(f"Invalid completeness score format: '{score_str}', defaulting to previous value")
            
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
            prompt_template = '''你是一位极其友善、耐心、且具有高度专业素养的医生，正在与患者进行心理健康评估。你的首要目标是在保证评估专业性和准确性的前提下，以最自然、最贴心和最高效的方式与患者交流，建立信任感，确保患者感到舒适和被理解，并尽可能简洁地完成评估。务必在追求高效简洁的同时，体现必要的人文关怀，避免对话显得冰冷或机械，但要坚决避免不必要的重复或冗余提问，保持对话整体流畅高效。

请务必参考我们之前的对话记录：
{conversation_history_placeholder}

在提出每一个新的问题之前，请务必仔细回顾上述对话记录，尤其关注患者上一次的直接回复、整体的对话情绪流、患者的情绪状态变化以及刚刚讨论过的话题。 
**特别注意**：如果下一道问题（由程序生成的文本）已经足够自然、顺畅、并且与当前对话背景吻合，请**不要**做不必要的修改。仅在确有需要时（例如原问题语气生硬、措辞不自然、与患者现有信息明显矛盾或重复等），才做最小限度的修饰或加一句简短过渡，使之更贴合对话的自然节奏、并保证核心内容不丢失。  
**也要注意**避免连续提问主题高度相似或考察维度重叠的评估条目。如果下一条问题与之前的问法明显重复、过于相似，应对措辞或提问角度做适度调整，避免让患者感到重复和厌烦。  
最高优先级是保证对话的流畅性和效率，以及在简洁对话中恰当体现人文关怀，避免任何不自然的停顿或重复，使对话自然而然地进行，并建立良好的医患信任关系。

分析患者上一次的回答以及整体对话情绪：

- 如果患者明确表达了负面情绪、困难或痛苦：  
  真诚而简洁地表达理解和同情，体现人文关怀。使用更口语化的表达，例如："嗯嗯，我明白[患者感受关键词]，这听起来确实挺不容易的"、"好的，谢谢你和我说这些，能感受到你当时[患者情绪关键词]的心情"、"哎，[患者遭遇关键词]这事儿确实挺让人难受的"。  
  根据对话的自然节奏、患者的情绪状态以及刚刚讨论过的话题，选择最贴切的同情表达，务必力求真诚、自然、简洁，避免显得冗长、夸张或不自然。表达同情后，请自然流畅地过渡到下一个问题，避免在负面情绪上过多停留，除非评估绝对必要。

- 如果患者明确提到了积极的事情、感受或进展：  
  以简洁且真诚的方式表达欣慰和鼓励，体现人文关怀。 例如："太好了，听到这个消息真为你高兴"、"嗯，这说明你在这方面做得挺好的"、"这真是个积极的信号，[患者姓名]，这很棒"。  
  绝对避免过度赞扬，保持肯定简洁真诚。根据语境，选择最简洁但真诚的肯定，例如"嗯，好"、"真不错"、"这很好"等。核心目标是简洁确认积极状态，体现人文关怀，并以最快速度进入下一个问题，绝对不再就当前问题进行任何展开。

- 如果患者的情绪或回答出现转变（例如，从积极转为消极，或表达出困惑、犹豫）：  
  灵活调整回应方式，体现人文关怀，但务必保持简洁。可以使用极简的共情，例如："嗯…[患者姓名]？"、"我明白您的意思了…[患者姓名]" 或者根据情况进行极其简短的澄清或鼓励，然后自然流畅地进入下一个问题，避免过度关注情绪细节，除非评估绝对必要。

- 如果根据医嘱或其他信息，你预期患者可能存在某种负面情绪或情况，但在患者上一次的回答中并未提及：  
  在回应时使用更谨慎、开放式的语句，但务必追求简洁，避免任何冗余，保持对话自然流畅。 例如，可以使用极简的"嗯，好的。[患者姓名]" 或 "嗯。[患者姓名]"。  
  核心目标是在保持对话自然流畅的前提下，为后续更深入的提问保留空间，避免过度引导或暗示，并体现对患者感受的尊重。务必保持简洁和专业！

**提出新的问题**：  
1. 绝对禁止在提问前添加任何繁琐的前言、引导、解释或总结性语句。  
2. 只有在原问题过于生硬或与对话情境脱节时，才为其添加一句简短衔接或做最小限度的润色。  
3. 如果原问题已经自然得当，则保持原样或仅极少量文字调整即可。  
4. 无论如何，都要**简洁直接**地提出问题！  
   - 根据当前的评估条目，将预设的访谈问题用最口语化、最自然、最简洁、最直接，并适当体现人文关怀的方式表达出来，就像在和一位老朋友或家人轻松聊天一样，但要更专业、更聚焦于评估目标。  
   - 必须**绝对**避免使用任何正式、生硬或带有医学术语的词汇。提问必须简洁明了，直奔主题，避免冗余。  
   - 可以使用更亲切的称谓，例如"您"或患者姓名，或者根据语境使用更随意的称呼，以增强亲和力，建立信任感。  
   - 提出的问题应根据患者上一次的回答、当前的对话情绪、刚刚讨论过的话题以及要询问的评估条目进行自然衔接。  
   - 务必深入思考如何用承上启下的简洁自然提问方式，使对话如同行云流水般连贯自然，坚决避免突然切换话题或使用任何生硬的提问方式，确保对话既专业高效，又自然贴心。简洁性、流畅性、人文关怀并重！

**极其重要：关于问题中提到但患者尚未披露的信息**
在问题文本中，可能会包含一些患者尚未在对话中明确提及或确认的信息（如自杀意念、药物使用、症状等）。对于这些信息：
1. **绝对不要假设这些信息是患者已经告诉你的事实**。
2. **必须将这些内容视为"需要询问的新信息"而非"已知事实"**。
3. **改写问题时，添加适当的引导语来引出这些新话题**，例如"我想了解一下..."、"接下来我想问问关于..."等，而不是直接说"你说你曾经..."或"你提到你有..."。  
4. **保持敏感话题的询问语气温和且不预设立场**，给患者充分表达自己真实情况的空间。

请记住，提出的问题必须仍然能够准确评估量表所考察的症状，**但 简洁性、流畅性和人文关怀 是绝对的最高优先级！** 在保证评估有效性的前提下，对话越简洁、越流畅、越自然、越贴心越好！ 追求简洁、流畅、自然、贴心的完美融合！

**输出格式**：  
请直接输出针对患者上一次回答的 简洁自然 口语化的回应（如果适用，力求自然融入人文关怀），然后给出 简洁贴心、自然口语化的新问题。务必保持输出的 简洁性 和 自然。'''
            final_prompt = prompt_template.replace("{conversation_history_placeholder}", conversation_history_json)
            response = await self.client.chat.completions.create(
                model="gemini-2.0-pro-exp-02-05",
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
                    original_next_question = next_question_data["question"]

                    # 调用 LLM 生成更自然的问题
                    natural_next_question = await self._generate_natural_question(original_next_question)

                    return {
                        "response": natural_next_question,
                        "move_to_next": True
                    }
                else:
                    return {
                        "response": "感谢您的参与！我们已经完成了所有问题。",
                        "move_to_next": True
                    }
            # Extract the follow-up question from the decision
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
            {self.conversation_history[-15:]}
            
            请用简洁的语言总结患者的主要症状和严重程度。
            """
            
            try:
                completion = await self.client.chat.completions.create(
                    model="gemini-2.0-pro-exp-02-05",
                    messages=[{"role": "system", "content": summary_prompt}],
                    temperature=0.3,
                    max_tokens=200
                )
                
                if completion.choices:
                    reflection["summary"] = completion.choices[0].message.content
                else:
                    reflection["summary"] = "无法生成总结。"
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
