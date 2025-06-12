import os
from lightrag import LightRAG, QueryParam
from lightrag.llm import gpt_4o_mini_complete,openai_complete_if_cache,hf_model_complete,custom_embedding
import numpy as np
from lightrag.utils import EmbeddingFunc, compute_args_hash
from transformers import AutoModel, AutoTokenizer
import torch
class MetaBuffer:
    def __init__(self,llm_model,embedding_model,api_key=None,base_url="",rag_dir='./test'):
        self.api_key = api_key
        self.llm = llm_model
        self.embedding_model = embedding_model
        self.base_url = base_url
        if not os.path.exists(rag_dir):
            os.mkdir(rag_dir)
        self.rag = LightRAG(
        working_dir= './test',
        llm_model_func=self.llm_model_func,  # Use Hugging Face model for text generation
        embedding_func=EmbeddingFunc(
            embedding_dim=768,
            max_token_size=512,
            func=self.embedding_func
        )
    )
        
       
    async def llm_model_func(
        self, prompt, system_prompt=None, history_messages=[], **kwargs
    ) -> str:
        return await openai_complete_if_cache(
            self.llm,
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            api_key=self.api_key,
            base_url=self.base_url,
            **kwargs
        )
    async def embedding_func(self, texts: list[str]) -> np.ndarray:
        res =  await custom_embedding(
            texts,
            model_name= "/home/zemelee/models/bge-base-en-v1.5",
        )
        print("Embedding shape:", res.shape)
        return res

    
    def retrieve_and_instantiate(self,input):
        response = self.rag.query(input, param=QueryParam(mode="hybrid"))
        return response
    
    def dynamic_update(self,thought_template):
        # 判断与MetaBuffer中最相关的模板是否有差别
        prompt = """
            Find most relevant thought template in the MetaBuffer according to the given thought template, and Determine whether there is a fundamental difference in the problem-solving approach between this and the most similar thought template in MetaBuffer. If there is, output "True." If there is no fundamental difference, or if the two thought templates are highly similar, output "False."
        """
        # 原是问题的思想模板
        input = prompt + thought_template
        # Perform naive search
        response = self.rag.query(input, param=QueryParam(mode="hybrid"))
        print(response) # 查找思维模板中最相似的思维模板，并根据差异性判断是否更新MetaBuffer
        if self.extract_similarity_decision(response):
            print('MetaBuffer Updated!')
            self.rag.insert(thought_template)
        else:
            print('No need to Update!')

        
    def extract_similarity_decision(self,text):
        """
        This function takes the input text of an example and extracts the final decision
        on whether the templates are similar or not (True or False).
        """
        # Convert the text to lowercase for easier matching
        text = text.lower()
        
        # Look for the conclusion part where the decision is made
        if "false" in text:
            return False
        elif "true" in text:
            return True
        else:
            # In case no valid conclusion is found
            raise ValueError("No valid conclusion (True/False) found in the text.")
        
        
    