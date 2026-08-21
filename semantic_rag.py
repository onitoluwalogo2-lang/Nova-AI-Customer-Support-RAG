from openai import OpenAI
import numpy as np
import json

client = OpenAI()

SIMILARITY_THRESHOLD = 0.40

documents = {
    "return_policy": "Eligible products can be returned within 30 days of delivery.",
    "shipping_policy": "Standard shipping takes 3-5 business days.",
    "warranty_policy": "Nova Camera comes with a 2-year warranty.",
    "payment_policy": "Nova accepts Visa, Mastercard, and other supported payment methods."
}


def create_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )

    return response.data[0].embedding


def cosine_similarity(vector_a, vector_b):
    a = np.array(vector_a)
    b = np.array(vector_b)

    return np.dot(a, b) / (
        np.linalg.norm(a) * np.linalg.norm(b)
    )


import json

with open("embeddings.json", "r") as file:
    document_embeddings = json.load(file)

print("Document embeddings loaded.")

print("\n==============================")
print("       NOVA AI SUPPORT")
print("==============================")
print("Type 'exit' to end the conversation.")


while True:

    customer_question = input("\nCustomer: ")

    if customer_question.lower() in ["exit", "quit"]:
        print("\nNova: Thanks for contacting Nova Support. Goodbye!")
        break

    query_embedding = create_embedding(customer_question)

    results = []

    for name, embedding in document_embeddings.items():
        score = cosine_similarity(query_embedding, embedding)
        results.append((score, name))

    results.sort(reverse=True)

    top_k = 2

    best_score, best_document = results[0]

    if best_score < SIMILARITY_THRESHOLD:
        print("\nNova's answer:")
        print("----------------")
        print("I'm sorry, but I don't have enough information in my knowledge base to answer that question.")
        continue

    retrieved_documents = results[:top_k]

    retrieved_context = "\n".join(
        documents[name]
        for score, name in retrieved_documents
    )


    prompt = f"""
    You are Nova, a customer support assistant.

    Answer the customer's question using only the knowledge provided below.

    Knowledge:
    {retrieved_context}

    Customer question:
    {customer_question}

    Give a clear and concise answer.
    """

    response = client.responses.create(
        model="gpt-4o-mini",
        input=prompt
    )

    print("\nTop retrieved documents:")
    print("------------------------")

    for score, name in retrieved_documents:
        print(f"{name}: {score:.4f}")

    print("\nNova's answer:")
    print("----------------")
    print(response.output_text)