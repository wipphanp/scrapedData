from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

import warnings
warnings.filterwarnings("ignore")


# Step1 : Load Raw PDFs

#Extract data from the PDF

DATA_PATH = './data/'

# def load_pdf(data):
#     loader = DirectoryLoader(data,
#                     glob="*.pdf",
#                     loader_cls=PyPDFLoader)
    
#     documents = loader.load()

#     return documents

# documents = load_pdf(data=DATA_PATH)
# print("\n \nLength of PDF Documents: ", len(documents))



def load_text_files(directory):
    """
    Loads text files from a directory using LangChain's DirectoryLoader.

    Args:
        directory (str): The path to the directory.

    Returns:
        List[Document]: A list of LangChain Document objects, or None if an error occurs.
    """

    print("Loading Text Files...", directory)
    try:
        text_loader_kwargs={'autodetect_encoding': True}
        loader = DirectoryLoader(directory, glob="*.txt", loader_cls=TextLoader, loader_kwargs=text_loader_kwargs)#glob to include subdirectories
        documents = loader.load()
        return documents
    
    except FileNotFoundError:
        print(f"Error: Directory '{directory}' not found.")
        return None
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None
    
documents = load_text_files(DATA_PATH)
print("\n \nLength of Text Documents: ", len(documents))


if documents:
    for doc in documents:
        print(f"Source: {doc.metadata['source']}")
        #print(f"Page Content: {doc.page_content}")
        print("-" * 20)


# Step2 : Create chunks 
#   
def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 50)
    text_chunks = text_splitter.split_documents(extracted_data)

    return text_chunks

text_chunks = text_split(extracted_data=documents)
print("length of my chunk:", len(text_chunks))

# Step3 : Create Vector Embeddings

def get_embedding_model():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embedding_model

embedding_model = get_embedding_model()

# Step4 : Store Embeddings in FAISS
DB_FAISS_PATH="vectorestore/db_faiss"
db=FAISS.from_documents(text_chunks, embedding_model)   
db.save_local(DB_FAISS_PATH)
