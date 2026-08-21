from openai import OpenAI
import json

client = OpenAI()


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


print("Creating document embeddings...")

document_embeddings = {}

for name, text in documents.items():
    print(f"Embedding: {name}")
    document_embeddings[name] = create_embedding(text)


with open("embeddings.json", "w") as file:
    json.dump(document_embeddings, file)


print("\nEmbeddings saved to embeddings.json")