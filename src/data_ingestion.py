from __future__ import annotations
import os
import sys
import json
import uuid
import hashlib
import shutil
from pathlib import Path
from typing import Iterable, List, Optional, Dict, Any
import fitz
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_community.vectorstores import FAISS

from langchain_community.vectorstores import FAISS
from logger import global_logger as log
from exceptions.custom_exception import DocumentPortalException
from utils.model_loader import ModelLoader
from utils.file_IO import *
import re


Supported_Extensions= (".pdf", ".txt", ".docx")

class ChatIngestor:
    def __init__ (self, temp_base:str = "data", 
                  faiss_base:str = "faiss_index",
                  use_session_dirs:bool = True,
                  session_id:Optional[str] = None,):
       try:
            self.model_loader= ModelLoader()
            self.use_session = bool (use_session_dirs)

            self.session_id= session_id or generate_session_id()

            self.temp_base= Path(temp_base)
            self.temp_base.mkdir(parents=True, exist_ok=True)

            self.faiss_base= Path(faiss_base)
            self.faiss_base.mkdir(parents=True, exist_ok=True)

            
            self.temp_dir = self._resolve_dir(self.temp_base) # Here _resolve_dir function will explain later under this class
            self.faiss_dir = self._resolve_dir(self.faiss_base) # This folder will be passsed and used when call object of class FaissManager to load or create vector store
            
            log.info (f"Chat Ingestor Initialized", 
                    session_id= self.session_id,
                    temp_dir=  str(self.temp_dir),
                    faiss_dir=str(self.faiss_dir))
       except Exception as e:
            log.error("Failed to initialize ChatIngestor", error=str(e))
            raise DocumentPortalException("Initialization error in ChatIngestor", e) from e
        
    def _resolve_dir(self, base:Path):
        """If Multiple Session or use_session_dir or self.session_dir = True"""
        # user or session base seperate folder will be formed
        if self.use_session:
            d= base/self.session_id
            d.mkdir(parents=True, exist_ok=True) #faiss_base or faiss_index/abhishek2007
            return d
        return base # fallback: "faiss_index/"
        """
            This Faiss_index will be passed in Class FaissManager, where it store files in following manner:

            faiss_index/
                └── session_20251019_101530_ab12cd34/abhishek2007
                        ├── index.faiss
                        ├── index.pkl
                        └── ingested_meta.json
            """
    

    def _split(self, docs:List[Document], chunk_size=1000, chunk_overlap=200):
        """This Function Split Documents into small chunks that will be used by bulit_in_retrieval function later"""
        splitter= RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks= splitter.split_documents(docs)
        log.info ("Documents splitted into chunks", chunks=len(chunks), chunk_size=chunk_size)
        return chunks


    def built_in_retrieval(self, uploaded_files:Iterable, *, chunk_size=1000, chunk_overlap=200):
       
        """This Function takes uploaded files in the form of Iterable(list, tupple, dictionary) and 
        than saved first and than loaded """
    # All files will be saved in self.temp_dir (define earlier inside __init__)
    # After Files has been saved, all Documents will be Loaded in Specified manner, provided by Langchain for Pdf, docs, txt

        try:
            paths= save_uploaded_files (uploaded_files, self.temp_dir) #Here self.temp_dir==target_dir:Path 
            docs= load_documents(paths)
            if not docs:
                raise ValueError("No Valid Document Loaded")
            
            #Now make Chunks with the help of _split function
            chunks= self._split(docs, chunk_size=chunk_size, chunk_overlap= chunk_overlap)

            #Seprerated Page_content and Metadata of Chunks
            texts = [c.page_content for c in chunks]
            meta_data = [c.metadata for c in chunks]

            #Lets Load Object of class FaissManager
            fm = FaissManager(self.faiss_dir, self.model_loader) 
            #Here self.faiss_dir ==self.index_dir that is described in class FaissManager
            try:
                vs = fm.load_or_create (texts=texts, metadatas=meta_data)
            except Exception:
                vs = fm.load_or_create (texts=texts, metadatas=meta_data)
            added = fm.add_docs(chunks)
            log.info("FAISS index updated", added=added, index=str(self.faiss_dir))
            return vs.as_retriever(search_type="similarity", search_kwargs={"k": 5})
        
        except Exception as e:
            log.error("Failed to build retriever", error=str(e))
            raise DocumentPortalException("Failed to build retriever", e) from e




        """This save_uploaded_files uses outside class ChatIngestor"""
def save_uploaded_files (uploaded_files:Iterable, target_dir:Path) ->List[Path]:
        """In this Function Evry File will be Saved in Folder, Here target_dir == self.temp_dir or self.temp_base"""
        try:
            target_dir.mkdir(parents= True, exist_ok=True)

            saved_files: List[Path]= []
            Supported_Extensions=[".pdf", ".txt", ".docx"]

            for uf in uploaded_files:
                name= getattr (uf, "name", "file")  # It will take name of uf(uploaded_file) and if name is absent than it will take "file"as name by default
                p=Path(name) # Make Path Object of name, so that we can easily access method of path to get extension of files
                ext = p.suffix.lower() # Get Extension like .pdf .docs .txt

                if ext not in Supported_Extensions:
                    log.warning ("Unsupported file Skipped", filename= name)
                    continue

                 #Now, Clean File Name with RegEx Function: Only alphanum, dash, underscore permit in file name
                safe_name = re.sub(r'[^a-zA-Z0-9_\-]', '_', p.stem).lower() # HERE p is Path(name)

                # Now Lets Convert this safe_name(file_name) into Unique ID

                fname= f"{safe_name}_{uuid.uuid4().hex[:6]}{ext}"

                #Now we can Overwrite these files into full Unique code, that even original fila_name will not appear
                
                # fname= f"{uuid.uuid4().hex[:8]}{ext}", but presently we use fname with unique code and filename

                # Now keep these Unique file inside folder self.target_dir and Prepares the exact path file will be stored

                out= target_dir/fname

                """This section is responsible for writing the uploaded file’s binary content to your local disk path (out)."""

                """In Some framework like FASTAPI and DJANGO, where Upload Source (like' uploaded file')
                 object has method .read(), but in Some Framework Upload Source (like 'Byte I/o', has method .getbuffer())"""
               
                with open (out, "wb") as f:  #Opens a file in binary write mode (wb = write bytes). (This creates a new file at path 'out' (or overwrites if it exists).)
                    if hasattr (uf, "read"):
                        f.write(uf.read())
                    else:
                        f.write (uf.getbuffer()) #fallback

                saved_files.append(out)
                log.info("File saved for ingestion", uploaded=name, saved_as=str(out))

            return saved_files  #Uploaded files has been saved and write in Disk
        except Exception as e:
            log.error("Failed to save uploaded files", error=str(e), dir=str(target_dir))
            raise DocumentPortalException("Failed to save uploaded files", e) from e

def load_documents (paths: Iterable[Path]) -> List[Document]:
     # Here Document is Class for storing a piece of text and associated metadata
     #from langchain_core.documents import Document
    """Load docs using appropriate loader based on extension."""
    docs:List[Document]=[]
    try:
        for j in paths:
            ext = j.suffix.lower()
            if ext == ".pdf":
                loader = PyPDFLoader(str(j))
            elif ext == ".docx":
                loader = Docx2txtLoader(str(j))
            elif ext == ".txt":
                loader = TextLoader(str(j), encoding="utf-8")
            else:
                log.warning("Unsupported extension skipped", path=str(j))
                continue
            docs.extend(loader.load())
        log.info("Documents loaded", count=len(docs))
        return docs

    except Exception as e:
        log.error("Failed loading documents", error=str(e))
        raise DocumentPortalException("Error loading documents", e) from e
    



class FaissManager:
    def __init__(self, index_dir: Path, model_loader: Optional[ModelLoader]= None):
        self.index_dir= Path(index_dir)
        self.index_dir.mkdir(exist_ok= True)
        self.meta_path= self.index_dir/"ingested_meta.json"
        self._meta:Dict[str:Any]= {"rows":{}}

        if self.meta_path.exists():
            try:
                text=self.meta_path.read_text (encoding= "utf-8")  or {"rows":{}} #It covert all meta data in Python Dictionary form
                self._meta= json.load(text) 

            except Exception:
                self._meta={"rows:{}"} #init the Empty One if it does not exist

        self.model_loader= model_loader or ModelLoader()
        self.emb = self.model_loader.load_embeddings()
        self.vs: Optional[FAISS]= None

    def _exist(self) -> bool:
        return (self.index_dir / "index.faiss").exists() and  (self.index_dir / "index.pkl").exists()
    
    @staticmethod
    def _fingerprint (text:str, md:Dict[str,Any])-> str:
        src= md.get ("source") or md.get("file_path")
        rid=  md.get("row_id")
        if src is not None:
            return f"{src} :: {'' if rid is None else rid}"
        return hashlib.sha256(text.encode("utf-8")).hexdigest()
    
    def save_meta(self):
        #This is Converted Python Dictionary In JSON FORMAT 
        return self.meta_path.write_text(json.dumps(self._meta, indent=2, ensure_ascii = False),encoding="utf-8")
    
    def add_docs(self, docs:List[Document]):
        if self.vs is None:
            raise RuntimeError("Call load or create before add_document_idempotent().")
        new_docs: List[Document] = []
        for d in docs:
            key= self._fingerprint(d.page_content, d.metadata or {})
            # d will iterate in self._meta["rows"] and check in "rows" inside ingested_meta.json.
            # if rows inside ingested_meta.json, already have source_id with respect to Row ID, it will skip to enter
            # If rows inside ingested_meta.json, doesnt have source_id with respect to Row ID, it will add that "key"(variable) or "d in docs" in row = True
            # After added that key (Assigned variable of 'd in docs'):::: d will be append or added in new_docs (Initialize earlier)
            
            if key in self._meta["rows"]:
                continue
            self._meta["rows"][key] = True
            new_docs.append(d)

        if new_docs:
            self.vs.add_documents (new_docs)  ### This add_documents is the Internal Method of Vector Store
            self.vs.save_local(str(self.index_dir))
            self.save_meta ()
            return len (new_docs)
        
    def load_or_create (self, texts: Optional[List[str]]=None, metadatas: Optional[List[dict]] = None):
         ## if we running first time then it will not go in this block
        if self._exist():
            self.vs=FAISS.load_local (
                str(self.index_dir),
                embeddings=self.emb,
                allow_dangerous_deserialization=True
            )
            return self.vs
        if not texts:
            raise DocumentPortalException (f"No Existing FAISS files and No Data to create one", sys)
        
        self.vs= FAISS.from_texts (
            texts=texts,    
            embedding=self.emb,
            metadatas=metadatas or []
        )
        self.vs.save_local(str(self.index_dir))
        return self.vs
   