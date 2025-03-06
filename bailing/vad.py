import os
import uuid
import wave
from abc import ABC, abstractmethod
import logging
from datetime import datetime
import collections

import numpy as np

import torch
from silero_vad import load_silero_vad, VADIterator

logger = logging.getLogger(__name__)


class VAD(ABC):
    @abstractmethod
    def is_vad(self, data):
        pass

    def reset_states(self):
        pass


class SileroVAD(VAD):
    def __init__(self, config):
        print("SileroVAD", config)
        self.model = load_silero_vad()
        self.sampling_rate = config.get("sampling_rate")
        self.threshold = config.get("threshold")
        self.min_silence_duration_ms = config.get("min_silence_duration_ms")
        self.vad_iterator = VADIterator(self.model,
                            threshold=self.threshold,
                            sampling_rate=self.sampling_rate,
                            min_silence_duration_ms=self.min_silence_duration_ms)
        logger.debug(f"VAD Iterator initialized with model {self.model}")

    @staticmethod
    def int2float(sound):
        """
        Convert int16 audio data to float32.
        """
        sound = sound.astype(np.float32) / 32768.0
        return sound

    def is_vad(self, data):
        try:
            # 检查数据是否是WAV文件（以'RIFF'开头）
            if data.startswith(b'RIFF'):
                # 跳过WAV文件头（44字节）
                data = data[44:]
                logger.debug("检测到WAV文件，已跳过文件头")
            
            # 确保数据长度是偶数（因为int16是2字节）
            if len(data) % 2 != 0:
                data = data + b'\x00'
                logger.debug(f"数据长度已补齐，从 {len(data)-1} 到 {len(data)} 字节")
            
            # 将字节数据转换为numpy数组
            audio_int16 = np.frombuffer(data, dtype=np.int16)
            audio_float32 = self.int2float(audio_int16)
            
            # 确定期望的chunk大小
            expected_chunk_size = 512 if self.sampling_rate == 16000 else 256
            
            # 将音频数据分成固定大小的chunk
            chunks = []
            for i in range(0, len(audio_float32), expected_chunk_size):
                chunk = audio_float32[i:i + expected_chunk_size]
                # 如果最后一个chunk不完整，补零
                if len(chunk) < expected_chunk_size:
                    chunk = np.pad(chunk, (0, expected_chunk_size - len(chunk)))
                chunks.append(chunk)
            
            # 处理每个chunk
            vad_results = []
            for chunk in chunks:
                try:
                    chunk_tensor = torch.from_numpy(chunk)
                    vad_output = self.vad_iterator(chunk_tensor)
                    if vad_output is not None:
                        vad_results.append(vad_output)
                except Exception as e:
                    logger.debug(f"Chunk处理错误: {e}")
                    continue
            
            # 如果任何一个chunk检测到语音活动，就返回True
            final_result = any(vad_results) if vad_results else False
            logger.debug(f"处理了 {len(chunks)} 个chunk，VAD结果: {final_result}")
            return final_result
            
        except Exception as e:
            logger.error(f"VAD 处理错误: {e}")
            logger.error(f"输入数据长度: {len(data)} 字节")
            return False

    def reset_states(self):
        try:
            self.vad_iterator.reset_states()  # Reset model states after each audio
            logger.debug("VAD states reset.")
        except Exception as e:
            logger.error(f"Error resetting VAD states: {e}")


def create_instance(class_name, *args, **kwargs):
    # 获取类对象
    cls = globals().get(class_name)
    if cls:
        # 创建并返回实例
        return cls(*args, **kwargs)
    else:
        raise ValueError(f"Class {class_name} not found")
