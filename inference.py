from bot_pipeline import BoT
import argparse
# import os
# os.environ['http_proxy'] = 'http://127.0.0.1:7890'
# os.environ['https_proxy'] = 'http://127.0.0.1:7890'

# 调用任意一个嵌入模型均可
embedding_model = 'BAAI/bge-base-en-v1.5' # TODO
# 调用KIMI大模型
llm_model = 'moonshot-v1-32k' # TODO
api_key = 'sk-' # TODO
base_url = 'https://api.moonshot.cn/v1' # TODO
rag_dir = './math'
prompt = "Solve the problem: A was born 6 years before B. A had a son at the age of 23. If B is now 31, how many years ago was A's son born?"
bot = BoT(
          user_input= prompt, 
          api_key = api_key,
          model_id = llm_model,
          embedding_model = embedding_model,
          base_url = base_url,
          )
bot.bot_inference()
