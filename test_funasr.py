#!/usr/bin/env python3
# coding = utf-8

import os
import sys
import wave
import logging
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bailing.asr import create_instance

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_audio_file(file_path):
    """检查音频文件"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"音频文件不存在: {file_path}")
        
    try:
        with wave.open(file_path, 'rb') as wf:
            # 检查音频格式
            channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            framerate = wf.getframerate()
            logger.info(f"音频格式: 声道数={channels}, 采样位数={sampwidth*8}bit, 采样率={framerate}Hz")
    except Exception as e:
        logger.error(f"检查音频文件时出错: {e}")
        raise

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("使用方法: python test_funasr.py <音频文件路径>")
        print("例如: python test_funasr.py test.wav")
        sys.exit(1)

    wav_file = sys.argv[1]
    
    try:
        # 检查音频文件
        check_audio_file(wav_file)
        
        # 配置FunASR
        config = {
            "model_dir": "models/SenseVoiceSmall",  # 模型目录
            "output_file": "tmp/"  # 输出目录
        }
        
        # 确保输出目录存在
        os.makedirs(config["output_file"], exist_ok=True)
        
        # 创建ASR实例并识别
        asr = create_instance('FunASR', config)
        logger.info(f"开始识别文件: {wav_file}")
        text, output_file = asr.recognizer(wav_file)
        
        if text:
            logger.info(f"识别结果: {text}")
            logger.info(f"输出文件: {output_file}")
        else:
            logger.error("识别失败，没有得到结果")
            
    except Exception as e:
        logger.error(f"处理音频时出错: {e}")
        sys.exit(1)
