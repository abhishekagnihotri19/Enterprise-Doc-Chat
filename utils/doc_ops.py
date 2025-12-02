from __future__ import annotations
from typing import Iterable, List
from pathlib import Path
from fastapi import UploadFile

class FastApiFileHandler:
    """Adapt FastAPI UploadFile -> .name + .getbuffer() API"""
    def __init__(self, uf: UploadFile):
        self._uf= uf
        self.name= uf.filename
    def getbuffer (self) ->bytes:
        self._uf.file.seek(0)
        return self._uf.file.read()



