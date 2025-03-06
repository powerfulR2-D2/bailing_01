from typing import Dict, List
import json

class InterviewScript:
    def __init__(self, script_path: str):
        """Initialize with the path to the interview script JSON file."""
        self.script = self._load_script(script_path)
        self.current_index = 0
        
    def _load_script(self, script_path: str) -> List[Dict]:
        """Load and validate the interview script."""
        with open(script_path, 'r') as f:
            script = json.load(f)
            
        # Validate script format
        for item in script:
            if not isinstance(item, dict):
                raise ValueError("Each script item must be a dictionary")
            if "question" not in item:
                raise ValueError("Each script item must have a 'question' field")
            if "time_limit" not in item:
                raise ValueError("Each script item must have a 'time_limit' field")
        
        return script
    
    def get_current_question(self) -> Dict:
        """Get the current question and its metadata."""
        if self.current_index >= len(self.script):
            return None
        return self.script[self.current_index]
    
    def move_to_next_question(self) -> Dict:
        """Move to the next question and return it."""
        self.current_index += 1
        return self.get_current_question()
    
    def get_remaining_time(self) -> int:
        """Get the remaining time allocation for the current question."""
        if self.current_index >= len(self.script):
            return 0
        return self.script[self.current_index]["time_limit"]
    
    def is_complete(self) -> bool:
        """Check if we've completed all questions in the script."""
        return self.current_index >= len(self.script)
