from dotenv import load_dotenv
load_dotenv()
import os
from sentence_transformers import SentenceTransformer
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import SentenceTransformerEmbeddings  # Import Sentence Transformer Embeddings
from langchain.prompts import PromptTemplate
from langchain.llms.base import LLM
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import logging
import google.generativeai as genai

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))



model = SentenceTransformer('hkunlp/instructor-large')
#e = model.encode("What is your refund policy?")
#print(e)
sentence_transformer_embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

vectordb_file_path ="faiss_index"

def create_vector_db():
    loader = CSVLoader(file_path='/Users/muhammadahmer/Documents/codebasics/codebasics_faqs.csv', source_column="prompt", encoding="ISO-8859-1")
    data = loader.load()
    vectordb = FAISS.from_documents(documents=data, embedding=sentence_transformer_embeddings)
    vectordb.save_local(vectordb_file_path)


def get_qa_chain():
    # Load the FAISS vector database
    vectordb = FAISS.load_local(vectordb_file_path, sentence_transformer_embeddings)

    # Create the retriever from the FAISS vector store
    retriever = vectordb.as_retriever(score_threshold=0.7)

    # Define the custom prompt for answering questions
    prompt_template = """Given the following context and a question, generate an answer based on this context only.
    In the answer try to provide as much text as possible from "response" section in the source document context without making much changes.
    If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.

    CONTEXT: {context}

    QUESTION: {question}"""

    # Create the prompt template
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    # Set the chain type kwargs
    chain_type_kwargs = {"prompt": PROMPT}

    # Define the custom GeminiLLM class
    class GeminiLLM(LLM):
        """Custom Gemini model wrapper for LangChain"""

        def _call(self, prompt: str, stop=None, run_manager=None, **kwargs) -> str:
            try:
                model = genai.GenerativeModel("gemini-1.5-flash")
                response = model.generate_content(prompt)
                logging.debug(f"Response: {response.text}")
                return response.text  # Ensure we return a plain string
            except Exception as e:
                logging.error(f"Error generating content: {e}")
                return "Error generating response"

        @property
        def _llm_type(self) -> str:
            return "Gemini"

    # Instantiate the Gemini model
    llm = GeminiLLM()

    # Set up the RetrievalQA chain with the retriever and LLM
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # Uses the simple retrieval approach
        retriever=retriever,
        return_source_documents=False  # Returns the documents used for retrieval
    )

    return chain


if __name__ == "__main__":
    #create_vector_db()
    chain = get_qa_chain()

    # Define the user query
    query = "Do you guys provide internship and also do you offer EMI payments?"

    # Execute the query using `.invoke()`
    response = chain.invoke({"query": query})

    # Print the response
    print("Answer:", response["result"])