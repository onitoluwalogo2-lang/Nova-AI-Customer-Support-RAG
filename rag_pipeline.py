from openai import OpenAI

client = OpenAI()


documents = {
    "return_policy": "Eligible products can be returned within 30 days of delivery.",
    "shipping_policy": "Standard shipping takes 3-5 business days.",
    "warranty_policy": "Nova Camera comes with a 2-year warranty.",
    "payment_policy": "Nova accepts Visa, Mastercard, and other supported payment methods."
}


retrieved_documents = [
    documents["return_policy"]
]


customer_question = "What's the refund window?"

context = "\n".join(retrieved_documents)


prompt = f"""
You are a customer support assistant.

Answer the customer's question using only the information provided below.

Knowledge:
{context}

Customer question:
{customer_question}
"""


response = client.responses.create(
    model="gpt-4o-mini",
    input=prompt
)


print("Nova's answer:")
print("----------------")
print(response.output_text)