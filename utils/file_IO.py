from __future__ import annotations
import uuid
from zoneinfo import ZoneInfo
from datetime import datetime


def generate_session_id(prefix:str="session") ->str:
    ist= ZoneInfo("Asia/Kolkata")
    return f"{prefix}_{datetime.now(ist).strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}"

