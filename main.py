import streamlit as st
from langchain_helper import get_qa_chain, create_vector_db  # Assuming these functions are in langchain_helper.py

# Title for the Streamlit app
st.title("Codebasics Q&A 🌱")

# Button to trigger the creation of the knowledge base
btn = st.button("Create Knowledgebase")

# Trigger create_vector_db when the button is pressed
if btn:
    st.write("Creating knowledgebase... Please wait.")
    create_vector_db()
    st.write("Knowledgebase created successfully!")

# Input field for the user to ask a question
question = st.text_input("Ask your question:")

# If the user enters a question
if question:
    # Get the QA chain
    chain = get_qa_chain()

    # Execute the query using .invoke() to get the answer
    response = chain.invoke({"query": question})

    # Display the answer in Streamlit
    st.header("Answer")
    st.write(response.get("result", "Sorry, I don't have an answer for that."))
