import os
from openai import OpenAI
import helperFunction as hf
# account for deprecation of LLM model
import datetime

client = OpenAI(
    # api_key=os.environ.get("OPENAI_API_KEY"),
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",
)

# Get the current date
current_date = datetime.datetime.now().date()

# Define the date after which the model should be set to "gpt-3.5-turbo"
target_date = datetime.date(2024, 6, 12)

llm_model = "microsoft/Phi-3-mini-4k-instruct-gguf"

response = hf.get_completion("What is 1+1?", client, llm_model)
print(response)

customer_email = """
Arrr, I be fuming that me blender lid \
flew off and splattered me kitchen walls \
with smoothie! And to make matters worse,\
the warranty don't cover the cost of \
cleaning up me kitchen. I need yer help \
right now, matey!
"""

style = """American English \
in a calm and respectful tone
"""

prompt = f"""Translate the text \
that is delimited by triple backticks 
into a style that is {style}.
text: ```{customer_email}```
"""

response = hf.get_completion(prompt, client, llm_model)

print(response)