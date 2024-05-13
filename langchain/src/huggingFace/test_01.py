from langchain_community.llms import HuggingFaceEndpoint
import os

# Set your Hugging Face API token 
huggingfacehub_api_token = os.environ['HUGGINGFACEHUB_API_TOKEN']

# Define the LLM
llm = HuggingFaceEndpoint(repo_id='tiiuae/falcon-7b-instruct', huggingfacehub_api_token=huggingfacehub_api_token)

# Predict the words following the text in question
question = 'Whatever you do, take care of your shoes'
output = llm.invoke(question)

print(output)