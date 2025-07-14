
import os
os.environ['USE_TF'] = '0'
from retriever import Retriever
from transformers import pipeline

CSV_PATH = r'C:\Users\morey\Downloads\Week 8\Training Dataset.csv'

def main():
    print('Loading retriever...')
    retriever = Retriever(CSV_PATH)
    print('Loading generative model...')
    generator = pipeline('text-generation', model='distilgpt2')
    print('RAG Q&A Chatbot Ready! Type your question (or "exit" to quit)')
    while True:
        query = input('You: ')
        if query.lower() == 'exit':
            break
        context = retriever.retrieve(query)
        prompt = f"Context: {' '.join(context)}\nQuestion: {query}\nAnswer: "
        response = generator(prompt, max_length=100, do_sample=True)[0]['generated_text']
        print('Bot:', response.split('Answer: ')[-1].strip())

if __name__ == '__main__':
    main()
