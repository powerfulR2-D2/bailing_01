from typing import List, Dict
from openai import AsyncClient
import logging
import json

class ReflectionModule:
    def __init__(self, openai_client: AsyncClient):
        """Initialize the reflection module."""
        self.client = openai_client

    async def generate_reflection(self, conversation_history: List[Dict], current_hamd_item_id: str = None, current_hamd_question: str = None) -> str:
        """Generate a reflection on the conversation history, focusing on HAMD assessment quality."""
        try:
            if not conversation_history:
                return "Response Quality: No conversation history available\nKey Points Covered: None\nMissing Information: Initial response needed\nIsabella's Performance: Not applicable\nSuggested Focus: Start the interview"

            # Include information about the current HAMD item for more focused reflection
            context_info = ""
            if current_hamd_item_id and current_hamd_question:
                context_info = f"The current HAMD item being discussed is '{current_hamd_item_id}' with the question: '{current_hamd_question}'. "

            messages = [
                {"role": "system", "content": f"You are an AI assistant expert in analyzing interview responses, specifically for Hamilton Depression Rating Scale (HAMD) assessments. Analyze the following conversation history to evaluate the quality of the patient's responses, Isabella's interviewing skills, and suggest improvements for the next steps. Focus on whether the key aspects of the current HAMD item were adequately covered. {context_info} Provide feedback on:\n"
                                            "- **Response Quality:** How clear, relevant, and comprehensive was the patient's last response in addressing the question?\n"
                                            "- **Key Points Covered:** What specific aspects or keywords related to the current HAMD item were mentioned by the patient?\n"
                                            "- **Missing Information:** What crucial information related to the current HAMD item is still missing from the patient's response?\n"
                                            "- **Isabella's Performance:** How effectively did Isabella ask the questions and follow up? Was her tone appropriate? Did she handle the interaction well?\n"
                                            "- **Suggested Focus:** What specific questions or topics should Isabella focus on next to gather more complete information for this HAMD item or the overall assessment?"},
                {"role": "user", "content": f"Conversation History:\n{json.dumps(conversation_history, ensure_ascii=False, indent=2)}"}
            ]

            try:
                completion = await self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    temperature=0.6,  # 可以适当降低温度以获得更集中的反馈
                    max_tokens=300,  # 增加 token 限制以容纳更详细的反馈
                )
                if completion.choices:
                    return completion.choices[0].message.content.strip()
                return ""
            except Exception as e:
                logging.error(f"Error in chat completion for reflection: {str(e)}")
                return ""

        except Exception as e:
            logging.error(f"Error in generate_reflection: {str(e)}")
            return ""