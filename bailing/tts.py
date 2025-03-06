import asyncio
import logging
import os
import subprocess
import time
import uuid
import wave
from abc import ABC, ABCMeta, abstractmethod
from datetime import datetime
import pyaudio
from pydub import AudioSegment
from gtts import gTTS

import ChatTTS
import torch
import torchaudio
import aiohttp
import base64
from typing import Dict, Any, Optional
import requests
import urllib3

logger = logging.getLogger(__name__)


class AbstractTTS(ABC):
    __metaclass__ = ABCMeta

    @abstractmethod
    def to_tts(self, text):
        pass


class GTTS(AbstractTTS):
    def __init__(self, config):
        self.output_file = config.get("output_file")
        self.lang = config.get("lang")

    def _generate_filename(self, extension=".aiff"):
        return os.path.join(self.output_file, f"tts-{datetime.now().date()}@{uuid.uuid4().hex}{extension}")

    def _log_execution_time(self, start_time):
        end_time = time.time()
        execution_time = end_time - start_time
        logger.debug(f"执行时间: {execution_time:.2f} 秒")

    def to_tts(self, text):
        tmpfile = self._generate_filename(".aiff")
        try:
            start_time = time.time()
            tts = gTTS(text=text, lang=self.lang)
            tts.save(tmpfile)
            self._log_execution_time(start_time)
            return tmpfile
        except Exception as e:
            logger.debug(f"生成TTS文件失败: {e}")
            return None


class MacTTS(AbstractTTS):
    """
    macOS 系统自带的TTS
    voice: say -v ? 可以打印所有语音
    """

    def __init__(self, config):
        super().__init__()
        self.voice = config.get("voice")
        self.output_file = config.get("output_file")

    def _generate_filename(self, extension=".aiff"):
        return os.path.join(self.output_file, f"tts-{datetime.now().date()}@{uuid.uuid4().hex}{extension}")

    def _log_execution_time(self, start_time):
        end_time = time.time()
        execution_time = end_time - start_time
        logger.debug(f"执行时间: {execution_time:.2f} 秒")

    def to_tts(self, phrase):
        logger.debug(f"正在转换的tts：{phrase}")
        tmpfile = self._generate_filename(".aiff")
        try:
            start_time = time.time()
            res = subprocess.run(
                ["say", "-v", self.voice, "-o", tmpfile, phrase],
                shell=False,
                universal_newlines=True,
            )
            self._log_execution_time(start_time)
            if res.returncode == 0:
                return tmpfile
            else:
                logger.info("TTS 生成失败")
                return None
        except Exception as e:
            logger.info(f"执行TTS失败: {e}")
            return None


class EdgeTTS(AbstractTTS):
    def __init__(self, config):
        self.output_file = config.get("output_file", "tmp/")
        self.voice = config.get("voice")

    def _generate_filename(self, extension=".wav"):
        return os.path.join(self.output_file, f"tts-{datetime.now().date()}@{uuid.uuid4().hex}{extension}")

    def _log_execution_time(self, start_time):
        end_time = time.time()
        execution_time = end_time - start_time
        logger.debug(f"Execution Time: {execution_time:.2f} seconds")

    async def text_to_speak(self, text, output_file):
        communicate = edge_tts.Communicate(text, voice=self.voice)  # Use your preferred voice
        await communicate.save(output_file)

    def to_tts(self, text):
        tmpfile = self._generate_filename(".wav")
        start_time = time.time()
        try:
            asyncio.run(self.text_to_speak(text, tmpfile))
            self._log_execution_time(start_time)
            return tmpfile
        except Exception as e:
            logger.info(f"Failed to generate TTS file: {e}")
            return None


class CHATTTS(AbstractTTS):
    def __init__(self, config):
        self.output_file = config.get("output_file", ".")
        self.chat = ChatTTS.Chat()
        self.chat.load(compile=False)  # Set to True for better performance
        self.rand_spk = self.chat.sample_random_speaker()

    def _generate_filename(self, extension=".wav"):
        return os.path.join(self.output_file, f"tts-{datetime.now().date()}@{uuid.uuid4().hex}{extension}")

    def _log_execution_time(self, start_time):
        end_time = time.time()
        execution_time = end_time - start_time
        logger.debug(f"Execution Time: {execution_time:.2f} seconds")

    def to_tts(self, text):
        tmpfile = self._generate_filename(".wav")
        start_time = time.time()
        try:
            params_infer_code = ChatTTS.Chat.InferCodeParams(
                spk_emb=self.rand_spk,  # add sampled speaker
                temperature=.3,  # using custom temperature
                top_P=0.7,  # top P decode
                top_K=20,  # top K decode
            )
            params_refine_text = ChatTTS.Chat.RefineTextParams(
                prompt='[oral_2][laugh_0][break_6]',
            )
            wavs = self.chat.infer(
                [text],
                params_refine_text=params_refine_text,
                params_infer_code=params_infer_code,
            )
            try:
                torchaudio.save(tmpfile, torch.from_numpy(wavs[0]).unsqueeze(0), 24000)
            except:
                torchaudio.save(tmpfile, torch.from_numpy(wavs[0]), 24000)
            self._log_execution_time(start_time)
            return tmpfile
        except Exception as e:
            logger.error(f"Failed to generate TTS file: {e}")
            return None


class ByteDanceTTS(AbstractTTS):
    def __init__(self, config: Dict[str, Any]):
        """
        初始化字节跳动文本转语音客户端
        
        Args:
            config: 包含初始化参数的配置字典
        """
        # 默认配置
        default_config = {
            "output_file": "tmp/",
            "appid": "2165408254",
            "access_token": "kBI174xO84Y8yFM8O7E20xJCpD7cumma",
            "cluster": "volcano_icl",
            "voice_type": "S_9cxCiref1",
            "host": "openspeech.bytedance.com",
            "user_uid": "388808087185088"
        }
        
        # 合并默认配置和传入配置
        self.config = {**default_config, **config}
        
        # 设置输出文件目录
        self.output_file = self.config.get('output_file', "tmp/")
        os.makedirs(self.output_file, exist_ok=True)
        
        # 当前说话者
        self.current_speaker = self.config.get('voice_type', "S_9cxCiref1")
        
        # 日志配置
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        
        # 配置日志处理器
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
    
    def _generate_filename(self, extension=".mp3"):
        """
        生成唯一的文件名
        """
        return os.path.join(self.output_file, f"tts-{datetime.now().date()}@{uuid.uuid4().hex}{extension}")

    def _log_execution_time(self, start_time):
        """
        记录执行时间
        """
        end_time = time.time()
        execution_time = end_time - start_time
        self.logger.debug(f"Execution Time: {execution_time:.2f} seconds")

    def to_tts(self, text):
        """
        同步方法，生成语音文件
        """
        # 生成唯一的临时文件名
        tmpfile = self._generate_filename(".mp3")
        start_time = time.time()
        
        try:
            # 如果文本长度超过限制，截断
            max_text_length = 1000
            if len(text) > max_text_length:
                self.logger.warning(f"文本长度超过 {max_text_length} 字符，将被截断。")
                text = text[:max_text_length]
            
            # 准备请求参数
            request_json = {
                "app": {
                    "appid": self.config['appid'],
                    "token": "access_token",
                    "cluster": self.config['cluster']
                },
                "user": {
                    "uid": self.config['user_uid']
                },
                "audio": {
                    "voice_type": self.current_speaker,
                    "encoding": "mp3",
                    "speed_ratio": 1.0,
                    "volume_ratio": 1.0,
                    "pitch_ratio": 1.0,
                },
                "request": {
                    "reqid": str(uuid.uuid4()),
                    "text": text,
                    "text_type": "plain",
                    "operation": "query",
                    "with_frontend": 1,
                    "frontend_type": "unitTson"
                }
            }
            
            # 发送请求
            header = {"Authorization": f"Bearer;{self.config['access_token']}"}
            api_url = f"https://{self.config['host']}/api/v1/tts"
            
            # 禁用 SSL 警告
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            
            # 发送请求
            resp = requests.post(
                api_url, 
                json=request_json, 
                headers=header, 
                verify=False,  # 禁用 SSL 验证
                timeout=30
            )
            
            # 检查响应
            resp.raise_for_status()
            data = resp.json().get('data')
            
            if not data:
                self.logger.error("No audio data in response")
                return None
            
            # 解码并保存音频
            audio_data = base64.b64decode(data)
            with open(tmpfile, 'wb') as file:
                file.write(audio_data)
            
            # 记录执行时间
            self._log_execution_time(start_time)
            
            return tmpfile
        
        except Exception as e:
            self.logger.error(f"TTS generation error: {e}")
            import traceback
            traceback.print_exc()
            return None

    def set_voice(self, voice_type: str):
        """
        设置语音类型
        """
        self.current_speaker = voice_type
        self.logger.info(f"Voice type set to: {voice_type}")

    def sample_random_speaker(self):
        """
        返回当前说话者类型
        """
        return self.current_speaker


def create_instance(class_name, *args, **kwargs):
    # 获取类对象
    cls = globals().get(class_name)
    if cls:
        # 创建并返回实例
        return cls(*args, **kwargs)
    else:
        raise ValueError(f"Class {class_name} not found")
