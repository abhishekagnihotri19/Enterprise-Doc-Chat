import os
import sys
import json
from utils.config_loader import load_config
from logger import global_logger as log
from exceptions.custom_exception import DocumentPortalException
from dotenv import load_dotenv
import json
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

class ApiManager:
    REQUIRED_KEYS= ["GOOGLE_API_KEY", "GROQ_API_KEY"]
    def __init__(self):
       # self.env= load_dotenv()
        load_dotenv()
        self.config= load_config()
        raw= os.getenv ("API_KEYS")
        self.api_keys={}
        if raw:
                try:
                    parsed=json.loads(raw)
                    
                    if not isinstance(parsed, dict):
                        raise ValueError ("API keys is not valid json object")
                    self.api_keys=parsed
                except Exception as e:
                    log.warning("Loaded API keys from API_keys as JSON", error=str(e))
        
        for key in self.REQUIRED_KEYS:
            if not self.api_keys.get(key):
                env_val= os.getenv(key)
                if env_val:
                    self.api_keys[key]=env_val
                    log.info(f"API keys and env variable {key}loaded from individual env successfully")

        # FInal Check
        missing_my_keys= [k  for k in self.REQUIRED_KEYS if not self.api_keys.get(k)]
        if missing_my_keys:
            log.error("missing required api key ", missing_keys=missing_my_keys)
            raise DocumentPortalException  ("Missing API keys", sys)

        log.info("API keys loaded", keys={k: v[:6] + "..." for k, v in self.api_keys.items()})
    def got_keys (self, key:str)->str:
        val=self.api_keys.get(key)
        if not val:
            raise KeyError(f"API key for {key} is missing")
        return val

class ModelLoader:
    def __init__(self):
        if os.getenv("ENV", "local").lower() != "production":
            load_dotenv()
            log.info ("Enviromental Variable or .env is loaded and Running in Local Machine")
        else:
            log.info ("Running in Production Mode")
        self.api_key_mgr= ApiManager()
        self.config = load_config()
        log.info ("Yaml config file is Loaded", config_keys=list[self.config.keys()])
    def load_embeddings (self):
        """Load Embedinng Model"""
        try:
            model_name= self.config["embedding_model"]["model"]
            log.info("Embedding Model is Loading")
            return GoogleGenerativeAIEmbeddings(model=model_name)
        except Exception as e:
            log.error("Error Loading Embedding Model", error=str(e))
            raise DocumentPortalException ("Failed to Load Embeding Model", sys)
    def load_llm(self):
        """Loading of LLM initiates"""
        api_key= self.api_key_mgr.got_keys("GOOGLE_API_KEY")
        if not api_key:
            raise DocumentPortalException("API key is missing")

        try:
            llm_model=self.config["llm"]["google"]["llm_name"]
            log.info("LLM is Loading")
            return ChatGoogleGenerativeAI(model=llm_model,
                                          api_key=api_key,
                                          temperature=0.2,
                                          max_output_tokens=1024)
        
        except Exception as e:
            log.error("Error Loading Embedding Model", error=str(e))
            raise DocumentPortalException ("Failed to Load Embeding Model", sys)
        

        #python -m utils.model_loader
if __name__=="__main__":
    loader= ModelLoader()
    #embedding=loader.load_embeddings()
    #print("Embedding Model are loaded")
    #result=embedding.embed_query("My name is Abhishek Agnihotri")
    #print(f"Embedding Result {result}")

    llm=loader.load_llm()
   
    result_llm= llm.invoke("What is Your Name?")
    result_lm= result_llm
    print (f"LLM Output is {result_lm}")            


            



