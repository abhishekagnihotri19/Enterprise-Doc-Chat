from pydantic import BaseModel, RootModel
from typing import List, Union
from enum import Enum

class PromptType(str, Enum):
    CONTEXTUALIZE_QUESION ="contextualize_question"
    CONTEXT_QA = "context_qa"

