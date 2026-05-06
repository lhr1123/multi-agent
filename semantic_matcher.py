from sentence_transformers import SentenceTransformer, util
from typing import List


class SemanticMatcher:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def get_match_score(self, required_skills: List[str], capabilities: List[str]) -> int:
        if not required_skills:
            return 1
        
        if not capabilities:
            return 1
        
        req_emb = self.model.encode(required_skills, convert_to_tensor=True)
        cap_emb = self.model.encode(capabilities, convert_to_tensor=True)
        
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
        
        req_emb = self.model.encode(required_skills, convert_to_tensor=True)
        cap_emb = self.model.encode(capabilities, convert_to_tensor=True)
        
        similarity_matrix = util.cos_sim(req_emb, cap_emb)
        return similarity_matrix.cpu().tolist()
