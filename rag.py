from openai import OpenAI
import numpy as np

client = OpenAI()


def create_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )

    return response.data[0].embedding


def cosine_similarity(vector_a, vector_b):
    a = np.array(vector_a)
    b = np.array(vector_b)

    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


text_a = "Eligible products can be returned within 30 days of delivery."

text_b = "What's the refund window?"

embedding_a = create_embedding(text_a)
embedding_b = create_embedding(text_b)

similarity = cosine_similarity(embedding_a, embedding_b)

print("Embedding A dimensions:", len(embedding_a))
print("Embedding B dimensions:", len(embedding_b))
print(f"Cosine similarity: {similarity:.4f}")