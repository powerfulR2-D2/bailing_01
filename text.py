import json
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_questions(questions_str):
    try:
        # 先找到所有完整的问题对象
        valid_questions = []
        current_depth = 0
        start_pos = questions_str.find('{')
        
        # 跳过第一个大括号（最外层的开始）
        questions_str = questions_str[start_pos + 1:]
        current_question = ""
        
        for char in questions_str:
            if char == '{':
                current_depth += 1
                if current_depth == 1:  # 开始一个新的问题对象
                    current_question = '{'
                else:
                    current_question += char
            elif char == '}':
                current_depth -= 1
                current_question += char
                
                if current_depth == 0:  # 一个问题对象结束
                    try:
                        # 尝试解析这个问题对象
                        question = json.loads(current_question)
                        valid_questions.append(question)
                        current_question = ""
                    except json.JSONDecodeError:
                        # 如果解析失败，说明遇到了不完整的问题
                        break
                
            elif current_depth > 0:  # 在问题对象内部
                current_question += char
        
        return valid_questions
    except Exception as e:
        logger.error(f"处理错误: {str(e)}")
        return []

# 测试用例1：完整的JSON
test_json_1 = None

# 测试用例2：不完整的JSON
test_json_2 = '''
{
  "questions": [
    {
      "id": "greeting",
      "question": "您好！我是 Isabella，接下来我将与您进行一次简短的交流，目的是为了更好地评估您的情绪状况。请您放心，我会尽力营造一个舒适的谈话氛围。您只需要根据最近一周左右的感受，如实地告诉我您的想法就可以了。",
      "type": "instruction"
    },
    {
      "id": "depressed_mood",
      "question": "最近心情怎么样？最近一周有没有觉得比较难过，或者提不起劲的时候？",
      "type": "open_ended",
      "expected_topics": [
        "情绪低落",
        "悲伤",
        "沮丧"
      ],
      "follow_up_questions": [
        "可以具体说说是什么让你觉得难过吗？",
        "这种提不起劲的感觉，通常会在什么时候比较明显呢？"
      ]
    },
    {
      "id": "guilt",
      "question": "好的，那最近有没有觉得自己好像做错了什么事情，或者觉得不值得拥有某些东西？",
      "type": "open_ended",
      "expected_topics": [
        "内疚感",
        "自责",
        "不配"
      ],
      "follow_up_questions": [
        "可以举个例子吗？",
        "这种感觉出现的频率高吗？"
      ]
    },
    {
      "id": "suicide",
      "question": "最近有没有觉得活着没什么意思，或者想到过不如就这样结束算了？",
      "type": "open_ended",
      "expected_topics": [
        "自杀意念",
        "死亡想法"
      ],
      "follow_up_questions": [
        "这种想法出现的时候，你会怎么排解呢？",
        "有没有更具体的计划或者想法？"
      ],
      "caution": true
    },
    {
      "id": "insomnia_initial",
      "question": "晚上睡觉怎么样？入睡是不是比平时要慢一些？",
      "type": "open_ended",
      "expected_topics": [
        "入睡困难",
        "失眠"
      ],
      "follow_up_questions": [
        "大概需要多长时间才能入睡呢？",
        "睡前会做些什么准备吗？"
      ]
    },
    {
      "id": "insomnia_middle",
      "question": "睡觉的时候容易醒吗？醒了之后还能很快再睡着吗？",
      "type": "open_ended",
      "expected_topics": [
        "睡眠不宁",
        "夜醒"
      ],
      "follow_up_questions": [
        "一般晚上会醒几次呢？",
        "醒来后通常会做些什么？"
      ]
    },
    {
      "id": "insomnia_late",
      "question": "早上会不会醒得比平时早？醒了之后就很难再睡着吗？",
      "type": "open_ended",
      "expected_topics": [
        "早醒",
        "早醒失眠"
      ],
      "follow_up_questions": [
        "通常会比预期早醒多少时间呢？",
        "醒来后会觉得很疲惫吗？"
      ]
    },
    {
      "id": "work_activities",
      "question": "对于平时的工作、学习或者比较喜欢的活动，最近还有兴趣吗？",
      "type": "open_ended",
      "expected_topics": [
        "工作兴趣",
        "活动兴趣",
        "动力"
      ],
      "follow_up_questions": [
        "有没有什么以前喜欢做的事情，现在不太想做了？",
        "是因为什么原因感觉没兴趣了呢？"
      ]
    },
    {
      "id": "retardation",
      "question": "有没有觉得脑子反应变慢了，或者做事情比平时更费力？",
      "type": "open_ended",
      "expected_topics": [
        "思维迟缓",
        "行动迟缓",
        "精力不足"
      ],
      "follow_up_questions": [
        "在哪些方面感觉反应变慢了呢？",
        "做哪些事情会觉得比较吃
'''

if __name__ == "__main__":
    print("测试用例1（完整JSON）:")
    valid_questions = validate_questions(test_json_1)
    print(f"有效问题数量: {len(valid_questions)}")
    for q in valid_questions:
        print(f"- {q['id']}: {q['question']}")
    
    print("\n测试用例2（不完整JSON）:")
    valid_questions = validate_questions(test_json_2)
    print(valid_questions)
    