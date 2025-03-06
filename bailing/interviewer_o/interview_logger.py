import os
import json
from datetime import datetime

class InterviewLogger:
    def __init__(self, base_dir="interview_logs"):
        self.base_dir = base_dir
        self.current_log = []
        self.session_id = None
        self.ensure_log_directory()
    
    def ensure_log_directory(self):
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)
    
    def start_new_session(self):
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_log = []
    
    def log_interaction(self, speaker, text, timestamp=None):
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        
        self.current_log.append({
            "timestamp": timestamp,
            "speaker": speaker,
            "text": text
        })
        
        # 实时保存到文件
        self.save_current_session()
    
    def save_current_session(self):
        if not self.session_id:
            return
        
        filename = f"interview_{self.session_id}.json"
        filepath = os.path.join(self.base_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({
                "session_id": self.session_id,
                "start_time": self.current_log[0]["timestamp"] if self.current_log else datetime.now().isoformat(),
                "interactions": self.current_log
            }, f, ensure_ascii=False, indent=2)
        
        return filepath
