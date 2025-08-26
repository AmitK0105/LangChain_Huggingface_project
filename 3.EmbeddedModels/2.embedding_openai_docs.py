from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding= OpenAIEmbeddings(model="text-embedding-3-large", dimensions=32)

documents=["Delhi is the capital of India",
            "Kolkata is the capital of WestBengal",
            "Mumbai is the capital of Maharashtra",
            "Paris is the capital of France"]

#result= embedding.embed_query("Delhi is the capital of India")

result= embedding.aembed_documents(documents)

print(str(result))