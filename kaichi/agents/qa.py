from typing import Dict, Optional
import time

class QAPair:
    """Question-Answer Pair Structure"""
    def __init__(self, question: str, concept: str, answer: Optional[str] = None):
        self.question = question
        self.concept = concept
        self.answer = answer
        self.timestamp = time.time()
        
    def to_dict(self) -> Dict:
        """Convert to dictionary format"""
        return {
            "question": self.question,
            "concept": self.concept,
            "answer": self.answer,
            "timestamp": self.timestamp
        }
        
    @classmethod
    def from_dict(cls, data: Dict) -> 'QAPair':
        """Create instance from dictionary"""
        pair = cls(data["question"], data["concept"])
        pair.answer = data.get("answer")
        pair.timestamp = data.get("timestamp", time.time())
        return pair

class QAManager:
    """Manager for QA operations"""
    def __init__(self, cache_size: int = 100):
        self.qa_pairs: Dict[str, QAPair] = {}
        self.cache_size = cache_size
        self.usage_count: Dict[str, int] = {}
        
    def add_pair(self, question: str, concept: str, answer: Optional[str] = None):
        """Add new QA pair"""
        if len(self.qa_pairs) >= self.cache_size:
            self._remove_least_used()
            
        self.qa_pairs[question] = QAPair(question, concept, answer)
        self.usage_count[question] = 0
        
    def get_pair(self, question: str) -> Optional[QAPair]:
        """Retrieve QA pair"""
        if question in self.qa_pairs:
            self.usage_count[question] += 1
            return self.qa_pairs[question]
        return None
        
    def update_answer(self, question: str, answer: str):
        """Update answer for existing question"""
        if pair := self.get_pair(question):
            pair.answer = answer
            pair.timestamp = time.time()
            
    def _remove_least_used(self):
        """Remove least used QA pair"""
        if not self.usage_count:
            return
            
        min_usage = min(self.usage_count.values())
        to_remove = next(k for k, v in self.usage_count.items() if v == min_usage)
        
        self.qa_pairs.pop(to_remove, None)
        self.usage_count.pop(to_remove, None)
        
    def to_dict(self) -> Dict:
        """Convert all QA pairs to dictionary"""
        return {
            q: pair.to_dict() 
            for q, pair in self.qa_pairs.items()
        }
        
    @classmethod
    def from_dict(cls, data: Dict, cache_size: int = 100) -> 'QAManager':
        """Create instance from dictionary"""
        manager = cls(cache_size)
        for q, pair_data in data.items():
            pair = QAPair.from_dict(pair_data)
            manager.qa_pairs[q] = pair
            manager.usage_count[q] = 0
        return manager