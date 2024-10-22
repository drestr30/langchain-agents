
import os
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain import hub

load_dotenv()

# Function to extract text from a .txt file
def extract_text_from_txt(txt_file):
    with open(txt_file, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

# Load the .txt document
document_path = r'C:\Users\davidre\Documents\langchain-agent\mortgage_broker\policies.txt'
doc_text = extract_text_from_txt(document_path)

# Split document into chunks based on paragraphs (double newlines)
texts = doc_text.split("\n\n")

# Initialize OpenAI API (ensure your OPENAI_API_KEY is set as an environment variable)
llm = AzureChatOpenAI(
    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    temperature=0)

# Create embeddings for text chunks
embeddings = AzureOpenAIEmbeddings(
    azure_deployment= os.environ['AZURE_EMBD_DEPLOYMENT_NAME'],
    api_key = os.environ['AZURE_EMBD_API_KEY'],
    azure_endpoint= os.environ['AZURE_EMBD_ENDPOINT'],
    api_version = os.environ['AZURE_EMBD_API_VERSION'])

# Store the document vectors in FAISS
vector_store = FAISS.from_texts(texts, embeddings)

# Create the retriever for the QnA system
retriever = vector_store.as_retriever(k=2)

# rag_prompt ="""You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. 
# If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
# Question: {question} 
# Context: {context} 
# Answer:
# """
# prompt = ChatPromptTemplate.from_messages([
#         ("system",rag_prompt),
#         ]
# )

prompt = hub.pull("rlm/rag-prompt")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


qa_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)


if __name__ == '__main__':
    res = qa_chain.invoke('How does your renewal process work, especially for customers switching from another lender? ')

    print(res)