from langchain_openai import ChatOpenAI
import requests
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Weaviate
import weaviate
from weaviate.embedded import EmbeddedOptions
import os
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms import HuggingFaceHub
from langchain.llms import HuggingFaceHub
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatHuggingFace
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from datasets import Dataset
from sentence_transformers import CrossEncoder
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)
import streamlit as st
from sentence_transformers import SentenceTransformer
from ragas.metrics import faithfulness, answer_correctness, answer_similarity
from ragas import evaluate
from typing import Sequence
import pandas as pd


class SentenceTransformerEmbeddings:
  def __init__(self, model_name: str):
      self.model = SentenceTransformer(model_name)

  def embed_documents(self, texts):
      return self.model.encode(texts, convert_to_tensor=True).tolist()

  def embed_query(self, text):
      return self.model.encode(text, convert_to_tensor=True).tolist()


class MistralParser():
  stopword = 'Answer:'
  parser = ''

  def __init__(self):
    self.parser = StrOutputParser()

  def invoke(self,query):
    ans = self.parser.invoke(query)
    return ans[ans.find(self.stopword)+len(self.stopword):].strip()


class RAGEval:
    '''
    WorkFlow:
    1. Call RAGEval()
    2. Call ground_truth_prep()
    3. Call model_prep()
    4. Call query()
    5. Call raga()
    '''
    best = 3
    parse = StrOutputParser()


    def __init__(self,file_path,url,vb_key,gpt_key, embed_model, cross_model):
        # self.vector_db(file_path,"https://chatgpt-db-z0lfxjds.weaviate.network","wVmxx6E57W2zJueEb8S1o3cJjwRiaAJEMkRA", embed_model)
        # os.environ["OPENAI_API_KEY"] = gpt_key
        #self.chatgpt = ChatOpenAI(model="gpt-4")
        #self.chat_model = self.chatgpt
        #self.parser = self.parse
        self.cross_model = cross_model
        self.template = """You are an assistant for question-answering tasks.
        Use the following pieces of retrieved context to answer the question.
        If you don't know the answer, just say that you don't know.
        Use two sentences maximum and keep the answer concise.
        Question: {question}
        Context: {context}
        Answer:
        """
        self.file_path = file_path
        self.url = url
        self.vb_key = vb_key
        self.embed_model = embed_model
        self.vector_db(self.file_path, self.url, self.vb_key, self.embed_model)

    def ground_truths_prep(self,questions): # questions is a file with questions
        self.ground_truths = [[s] for s in self.query(questions)]
        self.vector_db(self.file_path, self.url, self.vb_key, self.embed_model)

    def vector_db(self,file_path, url, api_key, embed_model): # file_path is the file of dataset
        # Read the content of the file
        with open(file_path, 'rb') as file:
            data = file.read()

        # Create a Document object
        documents = [Document(page_content=data)]
        # Initialize the text splitter
        text_splitter = CharacterTextSplitter(chunk_size=1330, chunk_overlap=50)
        # Split the documents into chunks
        chunks = text_splitter.split_documents(documents)

        WEAVIATE_CLUSTER=url
        WEAVIATE_API_KEY=api_key
        client=weaviate.Client(embedded_options=EmbeddedOptions())

        # Initialize the custom embedding model
        embedding_model = embed_model

        # Assuming you have a Weaviate client and documents prepared as `client` and `chunks`
        # Create the Weaviate vector store
        vectorstore = Weaviate.from_documents(
            client=client,
            documents=chunks,
            embedding=embedding_model,
            by_text=False
        )
        self.retriever=vectorstore.as_retriever()

    def model_prep(self,model,parser_choice=parse): # model_link is the link to the model
        self.chat_model = model
        self.parser = parser_choice

    # def rag_chain(self,model):
    #     # Define prompt template
    #     self.prompt = ChatPromptTemplate.from_template(self.template)
    #     self.ragchain=(
    #               {
    #                   "context":self.cross_model.rank(query=RunnablePassthrough(), documents=self.retriever,return_documents=True),
    #                   "question":RunnablePassthrough()
    #               }
    #               | self.prompt
    #               | model
    #     )

    def query(self,question):
        self.questions = question

        prior_context = [docs.page_content for docs in self.retriever.get_relevant_documents(self.questions)]
        c = self.cross_model.rank(
              query=question,
              documents=prior_context,
              return_documents=True
            )[:len(prior_context)-2]
        self.context = [i['text'] for i in c]

        self.answers = self.parser.invoke(
                self.chat_model.invoke(
                  self.template.format(
                    question=self.questions,
                    context=self.context)
                )
            )

        return self.answers

    def raga(self): # metric: 1 for Context_Precision / 2 for Context_Recall / 3 for Faithfulness / 4 for Answer_Relevancy
        data = {
            "question": self.questions,
            "answer": self.answers,
            "contexts": self.context,
            "ground_truth": self.ground_truths
        }
        dataset=Dataset.from_dict(data)
        result=evaluate(
            dataset=dataset,
            metrics=[
                context_precision,
                context_recall,
                faithfulness,
                answer_relevancy
            ]
        )
        df=result.to_pandas()
        return df

@st.cache_resource
def load_bi_encoder():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2",model_kwargs={"device": "cpu"})

@st.cache_resource
def load_embedding_model():
    return SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

@st.cache_resource
def load_cross_encoder():
    return CrossEncoder("cross-encoder/ms-marco-TinyBERT-L-2-v2", max_length=512, device="cpu")

@st.cache_resource
def load_chat_model():
    template = '''
    You are an assistant for question-answering tasks.
    Use the following pieces of retrieved context to answer the question accurately.
    If you don't know the answer, just say that you don't know.
    Question: {question}
    Context: {context}
    Answer:
    '''
    return HuggingFaceHub(
    repo_id="mistralai/Mistral-7B-Instruct-v0.1",
    model_kwargs={"temperature": 0.5, "max_length": 64,"max_new_tokens":512, "query_wrapper_prompt":template}
)

bi_encoder = load_bi_encoder()
embedding_model = load_embedding_model()
cross_encoder = load_cross_encoder()
chat_model = load_chat_model()

#file_path = 'software_data.txt'
file_path = "software_data.txt"
# HUGGINGFACEHUB_API_TOKEN = "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"
url = st.secrets["WEAVIATE_URL"]
v_key = st.secrets["WEAVIATE_V_KEY"]
gpt_key = st.secrets["GPT_KEY"]
os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets["HUGGINGFACEHUB_API_TOKEN"]

mistral_parser = MistralParser()

re = RAGEval(file_path,url,v_key,gpt_key,embedding_model,cross_encoder) # file_path,url,vb_key,gpt_key):
re.model_prep(chat_model, mistral_parser) # model details

st.title('RAG Bot')
st.subheader('Converse with our Chatbot')
st.text("Some sample questions to ask:")
st.markdown("- What are adjustment points in the context of using a microscope, and why are they important?")
st.markdown("- What does alignment accuracy refer to, and how is it achieved in a microscopy context?")
st.markdown("- What are alignment marks, and how are they used in the alignment process?")
st.markdown("- What is the alignment process in lithography, and how does eLitho facilitate this procedure?")
st.markdown("- What can you do with the insertable layer in Smart FIB?")

if 'messages' not in st.session_state:
    st.session_state.messages = []
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What's up?"):
    st.markdown(prompt)
    st.session_state.messages.append({"role":"user", "content":prompt})

    response = re.query(prompt)

    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
def reset_conversation():
  st.session_state.messages = []
  # st.session_state.chat_history = None

st.button('Reset Chat', on_click=reset_conversation)
