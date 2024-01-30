from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.vectorstores.utils import filter_complex_metadata
from langchain.chains import ConversationalRetrievalChain


class ChatAI:
    chat_history = []
    vector_store = None
    retriever = None
    chain = None

    def __init__(self):
        self.model = ChatOllama(model="mistral", temperature=0)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1024, chunk_overlap=100
        )
        self.prompt = PromptTemplate.from_template(
            """
            <s> [INST] You are an assistant for question-answering tasks. Use the following pieces of retrieved context 
            to answer the question. If you don't know the answer, just say that you don't know. Use three sentences
             maximum and keep the answer concise. [/INST] </s> 
            [INST] Question: {question} 
            Context: {context} 
            Answer: [/INST]
            """
        )

    def ingest(self, pdf_file_path: str):
        docs = PyPDFLoader(file_path=pdf_file_path).load()
        chunks = self.text_splitter.split_documents(docs)
        chunks = filter_complex_metadata(chunks)

        vector_store = Chroma.from_documents(
            documents=chunks, embedding=FastEmbedEmbeddings()
        )
        self.retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": 3,
                "score_threshold": 0.5,
            },
        )

        self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.model,
            chain_type="stuff",
            retriever=self.retriever,
            combine_docs_chain_kwargs={"prompt": self.prompt},
            return_source_documents=True,
            return_generated_question=True,
        )

    def ask(self, query: str):
        if not self.chain:
            return "Please, add a PDF document first."

        result = self.chain.invoke(
            {"question": query, "chat_history": self.chat_history}
        )
        self.chat_history.extend([(query, result["answer"])])

        print(f"\n\nChat Histry: {self.chat_history}")
        print(f"\n\nSource Documents: {result['source_documents']}")
        print(f"\n\nGenerated Question: {result['generated_question']}")

        return result["answer"]

    def clear(self):
        self.vector_store = None
        self.retriever = None
        self.chain = None
        self.chat_history = []
