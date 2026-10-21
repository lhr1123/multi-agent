from sentence_transformers import SentenceTransformer, util
import threading
from collections import OrderedDict
from torch import stack
from typing import List


class SemanticMatcher:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self._lock = threading.RLock()
        self._cache = OrderedDict()

    def _encode(self, texts):
        missing = list(dict.fromkeys(text for text in texts if text not in self._cache))
        if missing:
            embeddings = self.model.encode(missing, convert_to_tensor=True)
            self._cache.update(zip(missing, embeddings))
        result = [self._cache[text] for text in texts]
        for text in texts:
            self._cache.move_to_end(text)
        while len(self._cache) > 2048:
            self._cache.popitem(last=False)
        return stack(result)

    def get_task_match_score(self, subtask, agent) -> float:
        required = "\n".join([subtask.title, subtask.description, *subtask.required_skills])
        capability = "\n".join([agent.description, *agent.capabilities])
        with self._lock:
            req, cap = self._encode([required, capability])
            return max(0.0, min(10.0, float(util.cos_sim(req, cap)[0][0]) * 10))

    def get_match_score(self, required_skills: List[str], capabilities: List[str]) -> int:
        if not required_skills:
            return 1
        
        if not capabilities:
            return 1
        
        with self._lock:
            req_emb = self._encode(required_skills)
            cap_emb = self._encode(capabilities)
        
        sim_matrix = util.cos_sim(req_emb, cap_emb)
        
        max_sims = [max(row) for row in sim_matrix]
        
        avg_sim = sum(max_sims) / len(max_sims)
        
        return max(1, int(avg_sim * 10))
    
    def compute_similarity_matrix(
        self, 
        required_skills: List[str], 
        capabilities: List[str]
    ) -> List[List[float]]:
        if not required_skills or not capabilities:
            return [[1.0] * len(capabilities)] * len(required_skills)
        
        with self._lock:
            req_emb = self.model.encode(required_skills, convert_to_tensor=True)
            cap_emb = self.model.encode(capabilities, convert_to_tensor=True)
        
        similarity_matrix = util.cos_sim(req_emb, cap_emb)
        return similarity_matrix.cpu().tolist()
