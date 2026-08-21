import numpy as np


def cosine_similarity(vector_a, vector_b):
    a = np.array(vector_a)
    b = np.array(vector_b)

    return np.dot(a, b) / (
        np.linalg.norm(a) * np.linalg.norm(b)
    )


documents = {
    "return_policy": [1.0, 0.9, 0.1],
    "warranty_policy": [0.1, 0.2, 1.0],
    "shipping_policy": [0.8, 0.1, 0.2],
    "payment_policy": [0.1, 0.8, 0.1]
}

query = [0.9, 0.8, 0.1]

results = []

for name, embedding in documents.items():
    score = cosine_similarity(query, embedding)
    results.append((score, name))

results.sort(reverse=True)

top_k = 2

print("Top retrieved documents:")
print("------------------------")

for score, name in results[:top_k]:
    print(f"{name}: {score:.4f}")