import streamlit as st
import PyPDF2
import codecs
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import openai

# Set up OpenAI API credentials
openai.api_key = "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"


def convert_pdf_to_text(pdf_file, output_path):
    with codecs.open(output_path, 'w', encoding='utf-8') as output_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        output_file.write(text)


def create_vector_database(text_file):
    with codecs.open(text_file, 'r', encoding='utf-8') as file:
        texts = file.readlines()

    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(texts)

    return vectors, vectorizer, texts


def find_most_similar_question(question, vectorizer, vectors, texts):
    question_vector = vectorizer.transform([question])
    similarity_scores = vectors.dot(question_vector.T).toarray().flatten()
    most_similar_index = similarity_scores.argmax()
    return texts[most_similar_index]


def chatbot_gpt(prompt):
    response = openai.Completion.create(
        model="davinci",
        prompt=prompt,
        max_tokens=100,
        temperature=0.7,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )
    answer = response.choices[0].text.strip()
    return answer


def main():
    st.title("PDF Chatbot")

    # PDF Upload
    pdf_file = st.file_uploader("Upload PDF", type="pdf")

    if pdf_file is not None:
        # Convert PDF to Text
        text_file = "text_file.txt"
        convert_pdf_to_text(pdf_file, text_file)

        # Create vector database
        vectors, vectorizer, texts = create_vector_database(text_file)

        # Store vector database
        vector_db_file = "vector_db.pkl"
        with open(vector_db_file, "wb") as file:
            pickle.dump((vectors, vectorizer, texts), file)

        # User Question
        question = st.text_input("Enter your question")

        if question:
            # Find most similar question
            most_similar_question = find_most_similar_question(question, vectorizer, vectors, texts)

            st.info("Most Similar Question: " + most_similar_question)

            # Prompt for ChatGPT
            prompt = f"Question: {question}\nContext: {most_similar_question}"

            # Retrieve answer from ChatGPT API
            answer = chatbot_gpt(prompt)
            st.success("Answer: " + answer)


if __name__ == "__main__":
    main()
