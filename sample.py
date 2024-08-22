from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain_community.tools import ArxivQueryRun
from langchain_community.utilities import ArxivAPIWrapper
from langchain_google_genai import GoogleGenerativeAI
from langchain import hub
from dotenv import load_dotenv
import os
from langchain.agents import create_react_agent
from langchain.agents import AgentExecutor
import streamlit as st

load_dotenv()
## wikipedia tool
api_wrapper = WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=250)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper)
## Webbased tool
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=os.getenv("api_key"))

def create_custom_model(query,tool_name):
    loaders =  WebBaseLoader(query)
    docs = loaders.load()
    documents = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200).split_documents(docs)
    vectordb = FAISS.from_documents(documents,embeddings)
    retriever = vectordb.as_retriever()
    retrieval_tool = create_retriever_tool(retriever,tool_name,"Search for information from Custom_Tool")
    return retrieval_tool
## Arxiv tool
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1,doc_content_chars_max=250)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

st.title("RAG with Multiple Sources")

query=st.text_input("For custom model enter the website ")
tool_name=st.text_input("Enter the tool Name")
if query and tool_name:
    model = create_custom_model(query,tool_name)
    # tool list
    tools =[model, wiki, arxiv]

    ## loading llm and create template

    llm = GoogleGenerativeAI(model="gemini-pro", google_api_key=os.getenv("api_key"), temperature=0)

    os.environ["LANGCHAIN_API_KEY"] = os.getenv("Langchain_api_key")
    # Then, you can use the hub to pull the prompt
    prompt = hub.pull("hwchase17/react")
    # print(prompt.template)

    ### agent creation and agent executor creation 
    agent = create_react_agent(llm=llm,tools=tools,prompt=prompt)
    agent_executer = AgentExecutor(agent=agent,tools=tools,verbose=True)

    text = st.text_input("Enter the Query")
    if text:
        result = agent_executer.invoke({"input":text})
        st.subheader("Answer")
        st.write(result["output"])
# if __name__ == "__main__":
#     result = agent_executer.invoke({"input":"what is the capital of India?"})
#     print(result['output'])