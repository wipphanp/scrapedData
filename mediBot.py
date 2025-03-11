import streamlit as st
import os

from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

DB_FAISS_PATH="vectorestore/db_faiss"
@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db


def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt


# Load llm
def load_llm(huggingface_repo_id, HF_TOKEN):
    llm=HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.5,
        model_kwargs={"token":HF_TOKEN,
                      "max_length":1024}
    )
    return llm

def main():
    st.title("Ask my MediBot!")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message["role"]).markdown(message["content"])
    
    Prompt=st.chat_input("Pass your prompt here:")

    if Prompt:
        st.chat_message("user").markdown(Prompt)
        st.session_state.messages.append({"role": "user", "content": Prompt})

        CUSTOM_PROMPT_TEMPLATE = """
                Use the pieces of information provided in the context to answer the question at the end. If you don't know the answer, just say 
                that you don't know, don't try to make up an answer. Dont provide anything out of the context.

                Context: {context}
                Question: {question}

                Start the answer directly. No small talk please!

            """

        HF_TOKEN=os.environ.get("HF_TOKEN")
        HUGGINGFACE_REPO_ID="mistralai/Mistral-7B-Instruct-v0.3"
       

        try:
            vectorStore=get_vectorstore()
            if vectorStore is None:
                st.error("No vectorstore found : Failed to load the vectorstore")

            # Create QA Chain
            qa_chain = RetrievalQA.from_chain_type(
                llm=load_llm(huggingface_repo_id=HUGGINGFACE_REPO_ID,HF_TOKEN=HF_TOKEN),
                chain_type="stuff",                                    
                retriever=vectorStore.as_retriever(search_kwargs={"k": 3}), 
                return_source_documents=True,
                chain_type_kwargs={"prompt": set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
            )

            response = qa_chain.invoke({'query': Prompt})

            result=response['result']
            source_documents=response['source_documents']
            reslut_to_display=result+"\n\nSOURCE_DOCUMENTS:"+str(source_documents)

            #response= "Hello! How may I help you today?"

            st.chat_message("assistant").markdown(reslut_to_display)
            st.session_state.messages.append({"role": "assistant", "content": reslut_to_display})

        except Exception as e:
            st.error(f"An error occurred: {str()}")

        

if __name__ == "__main__":
    main()





