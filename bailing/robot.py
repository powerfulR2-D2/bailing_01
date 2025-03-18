import os
import sys
import time
import uuid
import wave
import queue
import asyncio
import logging
import datetime
import requests
import socketio
from gc import callbacks
import json
import threading
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
import argparse
from abc import ABC
from playsound import playsound
from bailing import (
    recorder,
    player,
    asr,
    tts
)
from bailing.dialogue import Message, Dialogue
from bailing.utils import is_interrupt, read_config, is_segment, extract_json_from_string
from plugins.registry import Action
from plugins.task_manager import TaskManager

from concurrent.futures import ThreadPoolExecutor, TimeoutError




from bailing.interviewer.interview_logger import InterviewLogger
from bailing.interviewer.question_generator import QuestionGenerator
from dotenv import load_dotenv
from openai import AsyncClient
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from typing import Optional
from bailing.interviewer.agent import InterviewerAgent
import asyncio
import socketio
import requests
load_dotenv()

logger = logging.getLogger(__name__)


app = FastAPI()

# Mount static files
static_path = os.path.join(os.path.dirname(__file__), "static")



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


class InterviewSession:
    def __init__(self,openai_client):
        self.openai_client = openai_client
        

        # 尝试读取文件内容
        """try:
            with open(self.default_script_path, 'r', encoding='utf-8') as f:
                content = f.read()
                logger.info(f"文件内容长度: {len(content)}")
        except Exception as e:
            logger.error(f"读取文件时发生错误: {e}")"""
        self.agent = None
        self.question_generator = QuestionGenerator(self.openai_client)
        
        self.logger = InterviewLogger()

    async def initialize_agent(self, patient_info):
        try:
            if patient_info:
                # Generate personalized questions based on patient info
                logging.info("Generating personalized questions...")
                logging.info(f"Patient info: {patient_info}")
                self.default_script_path = os.path.join(os.path.dirname(__file__), "interviewer\\temp_"+patient_info.get('scale')+"_script.json")
        
                logger.info(f"default_script_path: {self.default_script_path}")
                logger.info(f"文件是否存在: {os.path.exists(self.default_script_path)}")
                scale = patient_info.get('scale')
                patient_info_str=json.dumps(patient_info)
                questions = await self.question_generator.generate_questions(patient_info_str,scale)

                # 保存有效的问题到临时文件
                temp_script_path = os.path.join(os.path.dirname(__file__), "interviewer/temp_script.json")

                self.agent = InterviewerAgent(temp_script_path, self.openai_client)
            else:
                logging.info("Using default questions")
                self.agent = InterviewerAgent(self.default_script_path, self.openai_client)
                
            self.logger.start_new_session()
            return True
            
        except Exception as e:
            logging.error(f"Error initializing agent: {str(e)}")
            return False
"""# Create a dictionary to store active interview sessions
active_sessions = {}
# Get the session
session_id = "default"
session = InterviewSession()
# 初始化agent（可以传入患者信息，这里使用默认脚本）
success = session.initialize_agent()
  """  

class Robot(ABC):
    def __init__(self, config_file):
        config = read_config(config_file)
        self.audio_queue = queue.Queue()
        self.current_task = None
        self.task_lock = threading.Lock()
        self.async_lock = asyncio.Lock()

        # 尝试读取患者信息
        try:
            patient_info_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'server', 'patient_info.json')
            logger.info(f"尝试读取患者信息文件: {patient_info_path}")
            
            if os.path.exists(patient_info_path):
                with open(patient_info_path, 'r', encoding='utf-8') as f:
                    self.patient_info = json.load(f)
                logger.info(f"成功加载患者信息: {self.patient_info}")
            else:
                self.patient_info = None
                logger.warning("未找到患者信息文件，将使用默认设置")
        except Exception as e:
            self.patient_info = None
            logger.error(f"读取患者信息时发生错误: {e}")

        # 初始化其他组件
        self.asr = asr.create_instance(
            config["selected_module"]["ASR"],
            config["ASR"][config["selected_module"]["ASR"]]
        )

        self.openai_client = AsyncClient(
            api_key=config["LLM"][config["selected_module"]["LLM"]]["api_key"],
            base_url=config["LLM"][config["selected_module"]["LLM"]]["url"],
            timeout=120.0,
            max_retries=5,
        )
        self.tts = tts.create_instance(
            config["selected_module"]["TTS"],
            config["TTS"][config["selected_module"]["TTS"]]
        )


        self.player = player.create_instance(
            config["selected_module"]["Player"],
            config["Player"][config["selected_module"]["Player"]]
        )


        # 初始化线程池
        self.executor = ThreadPoolExecutor(max_workers=10)



        # 打断相关配置
        self.INTERRUPT = config["interrupt"]
        self.silence_time_ms = int((1000 / 1000) * (16000 / 512))  # ms

        # 线程锁
        self.chat_lock = False

        # 事件用于控制程序退出
        self.stop_event = threading.Event()

        self.callback = None

        self.speech = []

        self.task_queue = queue.Queue()
        self.task_manager = TaskManager(config.get("TaskManager"), self.task_queue)
        self.start_task_mode = config.get("StartTaskMode")
        
        # 专门用于 TTS 的队列
        self.tts_queue = queue.Queue(maxsize=1)  # 限制队列大小为1
        
        # 初始化 InterviewSession
        self.session = InterviewSession(self.openai_client)
        logger.info("InterviewSession 初始化完成")
        # 初始化 agent（传入患者信息）
        success = asyncio.run(self.session.initialize_agent(self.patient_info))
        
        if not success:
            logger.error("无法初始化 InterviewSession 的 agent")
        logger.info("InterviewSession 初始化完成")


        
    def listen_dialogue(self, callback):
        self.callback = callback
        


    def _tts_priority(self):
        logger.info("TTS 线程启动")
        def priority_thread():
            while not self.stop_event.is_set():
                try:
                    # 尝试获取最新的 TTS 任务
                    future = self.tts_queue.get(timeout=5)
                    
                    try:
                        # 获取 TTS 文件
                        logger.debug("准备获取 TTS 文件")
                        tts_file = future.result()
                        logger.debug(f"TTS 文件生成结果: {tts_file}")
                        
                        if tts_file and os.path.exists(tts_file):
                            logger.debug(f"TTS 文件存在: {tts_file}")
                            
                            # 通过回调通知
                            if self.callback:
                                logger.debug("准备调用回调函数")
                                self.callback({
                                    "role": "assistant",
                                    "type": "audio",
                                    "content": tts_file
                                })
                                logger.debug("回调函数调用完成")
                        else:
                            logger.error(f"TTS 文件无效: {tts_file}")
                    
                    except concurrent.futures.CancelledError:
                        logger.info("TTS 任务已被取消")
                    except TimeoutError:
                        logger.error("TTS 任务超时")
                    except Exception as e:
                        logger.error(f"TTS 处理错误: {e}")
                
                except queue.Empty:
                    time.sleep(0.1)
                except Exception as e:
                    logger.error(f"TTS 线程异常: {e}")
                    time.sleep(0.1)
        
        # 启动线程
        tts_priority = threading.Thread(target=priority_thread, daemon=True)
        tts_priority.start()

    def shutdown(self):
        """关闭所有资源，确保程序安全退出"""
        logger.info("Shutting down Robot...")
        self.stop_event.set()
        self.executor.shutdown(wait=True)
        #self.recorder.stop_recording()
        
        logger.info("Shutdown complete.")

    def start_recording_and_vad(self):
        """启动后台音频监听和处理"""
        # 启动音频监听线程
        audio_thread = threading.Thread(target=self._listen_audio_stream, daemon=True)
        audio_thread.start()
        
 
        # tts优先级队列
        self._tts_priority()

    def cancel_current_task(self):
        """取消当前正在执行的任务"""
        with self.task_lock:
            if self.current_task and not self.current_task.done():
                self.current_task.cancel()
                logger.info("已取消当前任务")

    def _listen_audio_stream(self):
        """后台监听音频流的线程函数"""
        sio = socketio.Client(ssl_verify=False)
        
        @sio.on('robot_audio_data')
        
        def on_audio_data(audio_data):
            logger.info("接收到音频数据:")
            logger.info(f"格式: {audio_data.get('format')}")
            logger.info(f"采样率: {audio_data.get('sampleRate')}")
            logger.info(f"时间戳: {audio_data.get('timestamp')}")
            logger.info(f"数据大小: {len(audio_data.get('audio', b''))} 字节")
            

            
            
            # 保存音频文件
            audio_dir = os.path.join(os.path.dirname(__file__), 'audio_files')
            os.makedirs(audio_dir, exist_ok=True)
            
            # 使用时间戳作为文件名
            timestamp = audio_data.get('timestamp', datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
            audio_path = os.path.join(audio_dir, f'robot_audio_{timestamp}.wav')
            
            # 保存音频数据
            with open(audio_path, 'wb') as f:
                f.write(audio_data.get('audio', b''))
            logger.info(f"robot_audio_data_音频文件已保存: {audio_path}")
            
            # 将完整的音频信息放入队列
            self.audio_queue.put(audio_data)
            
        sio.connect('https://localhost:5000')
        sio.wait()

    async def process_text(self, text):
        """处理文本的异步方法"""
        try:
            response_message = await self.chat_tool(text)
            if not self.current_task or not self.current_task.cancelled():
                await self.handle_response(response_message)
        except asyncio.CancelledError:
            logger.info("当前任务被取消")
        except Exception as e:
            logger.error(f"处理文本时出错: {e}")
        finally:
            self.chat_lock = False

    async def handle_response(self, response_message):
        """处理响应消息的异步方法"""
        try:
            if self.callback:
                self.callback({
                    "role": "assistant", 
                    "content": response_message['response'],
                    "need_confirm": response_message['need_confirm']
                })
                
                # 提交新的语音合成任务
                future = self.executor.submit(self.speak_and_play, response_message['response'])
                
                try:
                    old_future = self.tts_queue.get_nowait()
                    old_future.cancel()
                except queue.Empty:
                    pass
                
                try:
                    self.tts_queue.put_nowait(future)
                except queue.Full:
                    logger.error("TTS队列已满")

            if response_message.get('type') in ['image', 'draw']:
                if self.callback:
                    self.callback({
                        "role": "assistant",
                        "type": response_message.get('type'),
                        "content": response_message['image_path']
                    })
        except Exception as e:
            logger.error(f"处理响应时出错: {e}")

    def _duplex(self):
        """后台处理语音流的线程函数"""
        logger.debug("_duplex")
        
        # 识别到vad开始
        # 处理识别结果
        audio_data = self.audio_queue.get()
        if audio_data is None:
            return
        
        
        # 从字典中获取音频数据和相关信息
        audio_bytes = audio_data.get('audio', b'')
        if isinstance(audio_bytes, list):
            # 如果是整数列表，转换为字节列表
            audio_bytes = [bytes([b]) for b in audio_bytes]
        
        timestamp = audio_data.get('timestamp', datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
        
        # 保存音频文件
        audio_dir = os.path.join(os.path.dirname(__file__), 'duplex_audio')
        os.makedirs(audio_dir, exist_ok=True)
        
        # 使用时间戳作为文件名
        audio_path = os.path.join(audio_dir, f'duplex_{timestamp}.wav')
        
        # 保存音频数据到WAV文件
        with wave.open(audio_path, 'wb') as wf:
            wf.setnchannels(1)  # 单声道
            wf.setsampwidth(2)  # 16位采样
            wf.setframerate(16000)  # 16kHz采样率
            if isinstance(audio_bytes, list):
                wf.writeframes(b''.join(audio_bytes))
            else:
                wf.writeframes(audio_bytes)
        logger.info(f"_duplex方法中保存音频文件: {audio_path}")

        # 使用保存的WAV文件进行识别
        text, tmpfile = self.asr.recognizer(audio_path)

        # 检查识别结果
        if text is None:
            logger.debug("ASR识别结果为None，跳过处理")
            return
                
        if not text.strip():
            logger.debug("识别结果为空，跳过处理。")
            return

        logger.info(f"ASR识别结果: {text}")
        """       if self.current_task:
                        logger.info("取消当前任务")
                        self.cancel_current_task()  # 直接调用取消方法
        """
        if self.callback:
            self.callback({"role": "user", "content": str(text)})
                    
        # 取消当前任务（如果有）
        self.cancel_current_task()
        
        # 创建新的异步任务
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        with self.task_lock:
            self.current_task = loop.create_task(self.process_text(text))
        loop.run_until_complete(self.current_task)
        loop.close()
        return True
            

    def run(self):
        
        
        script_path = os.path.join("bailing", "interviewer", "temp_script.json")
        with open(script_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            initial_question=data["questions"][0]["question"]
        
        logger.info(f"初始问话：{initial_question}")
        # 发送开始录音信号
        try:
                response = requests.post('https://localhost:5000/start_main_recording', verify=False)
                if response.status_code == 200:
                    logger.info("已发送开始录音信号")
                else:
                    logger.error(f"发送开始录音信号失败: {response.text}")
        except Exception as e:
                logger.error(f"发送开始录音信号时发生错误: {str(e)}")

        # 更新对话
        tts_file = self.tts.to_tts(initial_question)
        #tts_file = None
        logger.debug(f"TTS 文件生成完毕{tts_file}")
        
        if self.callback:
            
            self.callback({
                "role": "assistant", 
                "content": initial_question,
                "need_confirm":False
            })
            self.callback({
                "role": "assistant",
                "type": "audio",
                "content": tts_file
            })
            
        
        try:
            self.start_recording_and_vad()  # 监听语音流
            logger.info("开始音频监听和处理...")
            while not self.stop_event.is_set():
                try:
                    self._duplex()  # 双工处理
                except queue.Empty:
                    # 队列为空是正常的，继续等待
                    continue
                except Exception as e:
                    logger.error(f"双工处理发生错误: {str(e)}", exc_info=True)
                    # 不要因为处理错误就退出循环
                    time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Received KeyboardInterrupt. Exiting...")
        except Exception as e:
            logger.error(f"音频处理主循环发生严重错误: {str(e)}", exc_info=True)
        finally:
            error_type = sys.exc_info()[0]
            if error_type is not None:
                logger.error(f"退出原因: {error_type.__name__}")
            logger.info("Shutting down Robot...")
            #self.shutdown()

    def speak_and_play(self, text):
        
        if text is None or len(text)<=0:
            logger.info(f"无需tts转换，query为空，{text}")
            return None
        tts_file = self.tts.to_tts(text)
        #tts_file = None
        if tts_file is None:
            logger.error(f"tts转换失败，{text}")
            return None
        logger.info(f"TTS 文件生成完毕{tts_file}")
        
        
        #if self.chat_lock is False:
        #    return None
        # 开始播放
        # self.player.play(tts_file)
        #return True
        # 播放语音
        #self.player.play(tts_file)
        return tts_file

    async def chat_tool(self, query):
        try:
            # 记录开始时间
            start_time = time.time()
            # 检查 agent 是否存在
            if self.session.agent is None:
                logger.error("Interview agent 未初始化")
                return {"response": "抱歉，系统出现了问题，无法生成回复。"}
            
            logger.info(f"开始处理查询: {query}")
            
            # 使用异步锁保护关键部分
            async with self.async_lock:
                # 创建一个任务
                self.current_task = asyncio.create_task(self.session.agent.generate_next_action(query))
                
                try:
                    # 等待任务完成，设置超时时间
                    llm_responses = await asyncio.wait_for(self.current_task, timeout=30.0)  # 30秒超时
                    logger.info(f"LLM 原始响应: {llm_responses}")
                except asyncio.TimeoutError:
                    logger.error("LLM 响应超时")
                    if not self.current_task.done():
                        self.current_task.cancel()
                    return {"response": "抱歉，响应时间过长，请稍后重试。"}
                except asyncio.CancelledError:
                    logger.info("任务被取消")
                    return {"response": "任务已被系统取消"}
                except Exception as e:
                    logger.error(f"LLM 调用失败: {str(e)}", exc_info=True)
                    if not self.current_task.done():
                        self.current_task.cancel()
                    return {"response": "抱歉，系统暂时无法处理您的请求。"}
                finally:
                    self.current_task = None
            
            # 记录响应生成时间
            end_time = time.time()
            logger.info(f"响应时间: {end_time - start_time:.2f} 秒")
            
            # 验证和处理响应
            if not isinstance(llm_responses, dict):
                logger.error(f"响应格式错误，期望 dict 类型，实际类型: {type(llm_responses)}")
                return {"response": "抱歉，系统返回格式错误。"}
            
            # 处理特殊类型响应
            if llm_responses.get('type') in ['image', 'draw']:
                logger.info("处理图片类型响应")
                return llm_responses
            
            # 获取响应内容
            content = llm_responses.get('response')
            if not content:
                logger.error(f"响应内容为空: {llm_responses}")
                return {"response": "抱歉，系统返回内容为空。"}
            
            logger.info(f"最终响应内容: {content}")
            return {
                "response": content,
                "need_confirm": llm_responses.get('need_confirm'),
                "type": llm_responses.get('type', 'text')
            }
            
        except Exception as e:
            import traceback
            error_stack = traceback.format_exc()
            logger.error(f"处理查询时发生错误: {query}\n错误类型: {type(e).__name__}\n错误信息: {str(e)}\n堆栈跟踪:\n{error_stack}")
            return {"response": "抱歉，系统处理出现错误，请稍后重试。"}

    async def chat(self, query):
        logger.debug("开始处理聊天消息")
        
        # 直接使用 await 调用异步方法
        response_message = await self.chat_tool(query)
        
        logger.info(f"生成响应消息: {response_message}")

        self.chat_lock = True
        
        try:
            logger.debug("开始处理聊天消息")
            
            # 更新对话
            if self.callback:
                
                logger.info("调用回调函数发送文本消息")
                self.callback({
                    "role": "assistant", 
                    "content": response_message['response'],
                    "need_confirm": response_message['need_confirm']
                })
                # 提交新的语音合成任务
                logger.debug("提交 TTS 任务")
                future = self.executor.submit(self.speak_and_play, response_message['response'])
                
                # 清空并放入最新任务
                try:
                    # 尝试取出并取消之前的任务
                    old_future = self.tts_queue.get_nowait()
                    old_future.cancel()
                    logger.debug("取消了之前的 TTS 任务")
                except queue.Empty:
                    logger.debug("队列为空，无需取消")
                
                # 放入新任务
                try:
                    self.tts_queue.put_nowait(future)
                    logger.debug("TTS 任务入队完成")
                except queue.Full:
                    logger.error("TTS 队列已满，无法放入新任务")
            if response_message.get('type') in ['image', 'draw']:
                logger.debug(response_message)
                logger.debug("回复为图片")
                if self.callback:
                    self.callback({
                        "role": "assistant",
                        "type": response_message.get('type'),
                        "content": response_message['image_path']
                    })
        except Exception as e:
            logger.error(f"语音处理错误: {e}")
        
        finally:
            self.chat_lock = False
        
        return True
if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description="百聆机器人")

    # Add arguments
    parser.add_argument('config_path', type=str, help="配置文件", default=None)

    # Parse arguments
    args = parser.parse_args()
    config_path = args.config_path

    # 创建 Robot 实例并运行
    robot = Robot(config_path)
    logger.info("Robot started")
    asyncio.run(robot.run())
