import sys
import os
from operator import itemgetter
from typing import List, Optional, Dict, Any
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS

from utils.model_loader import ModelLoader
from exceptions.custom_exception import DocumentPortalException
from logger import global_logger as log
from prompts.prompt import PromptRegistry
from langchain_core.runnables import RunnableLambda


from model.models import PromptType

class ConversationalRag: 
    
    #LCEL-based Conversational RAG with lazy retriever initialization.

#     Usage:
#         rag = ConversationalRAG(session_id="abc")
#         rag.load_retriever_from_faiss(index_path="faiss_index/abc", k=5, index_name="index") 
#         answer = rag.invoke("What is ...?", chat_history=[])
    
    def __init__(self, session_id : Optional[str], retriever=None):
            try:
                  self.session_id=session_id
                  self.model_loader= ModelLoader()

                  self.llm= self.model_loader.load_llm()
                  if not self.llm:
                        raise ValueError("LLM Could Not be loaded")
                  log.info("LLM Loaded Successfully", session_id= self.session_id)
                  self.contextualize_prompt: ChatPromptTemplate = PromptRegistry[PromptType.CONTEXTUALIZE_QUESION.value]
                  self.qa_prompt: ChatPromptTemplate = PromptRegistry [PromptType.CONTEXT_QA.value]

            # Lazy Pieces
                  self.retriever= retriever
                  self.chain = None
                  if self.retriever is not None:
                        self._build_lcel_chain()
                  #log.info("Coversational RAG Initilaized", session_id=self.session_id)
                  log.info("Conversational RAG initialized", extra={"session_id": self.session_id})
           
            except Exception as e:
                  log.error("Failed to Initialize RAG Conversation", error=str(e))
                  raise DocumentPortalException("Failed to convert RAG conversation", sys)
            
      
    def load_retriever_from_faiss (self, index_path: str, k: int = 5, index_name: str = "index", 
                                   search_type: str = "similarity",
                                     search_kwargs: Optional[Dict[str,Any]] = None,):
        
        """Load FAISS vectorstore from disk and build retriever + LCEL chain."""
             
        try:
              if not os.path.isdir(index_path):
                   raise FileNotFoundError (f"FAISS Index directory not found: {index_path}")
              embeddings = self.model_loader.load_embeddings()
              vectorstore = FAISS.load_local(
                   index_path,
                   embeddings = embeddings,
                   index_name = index_name,
                   allow_dangerous_deserialization = True, # ok if you trust the index

              )

              if search_kwargs is None:
                   search_kwargs= {"k": k}
              self.retriever = vectorstore.as_retriever(
                   search_type=search_type, search_kwargs=search_kwargs
              )
              self._build_lcel_chain()

              log.info(
                "FAISS retriever loaded successfully",
                index_path=index_path,
                index_name=index_name,
                k=k,
                session_id=self.session_id,
            )
              return self.retriever

        
        except Exception as e:
          log.error("Failed to load retrieval from FAISS", error = str(e))
          raise DocumentPortalException (f"Failed to load Conversation RAG", sys)
    
    def invoke (self, user_input:str, chat_history : Optional[List[BaseMessage]] = None) -> str:
         """Invoke LCEL Pipeline"""
         try:
              if self.chain is None:
                   raise DocumentPortalException(f"RAG chain Not initializa, call load_retriever_from_faiss(), before invoke", sys)
              chat_history = chat_history or []
              payload = {"input": user_input, "chat_history": chat_history}

              answer = self.chain.invoke(payload)

              if not answer:
                   log.warning("No Answer generated", user_input=user_input, session_id = self.session_id)
              log.info(
                "Chain invoked successfully",
                session_id = self.session_id,
                user_input = user_input,
                answer_preview = str(answer)[:150],
            )
              return answer
         except Exception as e:
              log.error("Failed to invoke ConversationalRAG", error=str(e))
              raise DocumentPortalException("Invocation error in ConversationalRAG", sys)
         
    def load_llm(self):
         try:
            self.llm = self.model_loader.load_llm()
            if not self.llm:
                raise ValueError("LLM could not be loaded")
            log.info("LLM loaded successfully", session_id = self.session_id)
            return self.llm
         except Exception as e:
            log.error("Failed to load LLM", error=str(e))
            raise DocumentPortalException("LLM loading error in ConversationalRAG", sys)
    
    @staticmethod
    #For each doc d, extract its text content. If page_content does not exist, use the string version of the doc.
    def _format_docs(docs) -> str:
         return "\n\n".join (getattr(d, "page_content", str(d)) for d in docs)
    
    
    def _build_lcel_chain (self):
         try:
              if self.retriever is None:
                   raise DocumentPortalException (f"No retriever set before building chain", sys)
              
                # 1) Rewrite user question with chat history context
              
              question_rewriter = (
                   {"input": itemgetter("input"), "chat_history": itemgetter("chat_history")}
                   | self.contextualize_prompt
                   | self.llm
                   | StrOutputParser()
                   
                              )
              
       
               # 2) Retrieve docs for rewritten question

              retrieve_docs= question_rewriter | self.retriever | ConversationalRag._format_docs

              # 3) Answer using retrieved context + original input + chat history

              self.chain = ( 
                  {
                   "context": retrieve_docs,
                   "input": itemgetter("input"),
                   "chat_history": itemgetter("chat_history"),
                  }
                  | self.qa_prompt
                  | self.llm
                  | StrOutputParser()
               )

              log.info("LCEL graph built successfully", session_id=self.session_id)
         except Exception as e:
            log.error("Failed to build LCEL chain", error=str(e), session_id=self.session_id)
            raise DocumentPortalException("Failed to build LCEL chain", sys)

