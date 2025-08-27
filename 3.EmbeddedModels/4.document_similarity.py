from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

embedding= OpenAIEmbeddings(model= "text-embedding-3-large", dimensions=300)

documents=["Virat Kohli is an Indian cricketer known for his aggressive batting and leadership",
           "MS Dhoni is a former Indian crickter known for his calmness, on field strategy and finishing skill",
           "Sachin Tendulkar, also known as 'God of cricket' holds many batting records",
           "Rohit Sharma known for his elegance batting and record breaking double centuries"]

query= "tell me about Virat Kohli"

#Documents embedding generation

doc_embedding= embedding.embed_documents(documents)

query_embedding= embedding.embed_query(query)

#print(cosine_similarity([query_embedding], doc_embedding))

score= cosine_similarity([query_embedding], doc_embedding)[0]

print(list(enumerate(score)))

print(sorted(list(enumerate(score))), key= lambda x:x[1])

# after sorting getting the last element

print(sorted(list(enumerate(score))), key= lambda x:x[1])[-1]

# save index and score in two variables

index, score= sorted(list(enumerate(score)), key= lambda x:x[1])[-1]

print(query)
print(documents[index])
print("Similarity score is :", score)