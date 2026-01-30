from datetime import datetime, timezone
from typing import Optional
from uuid import UUID, uuid4
from sqlmodel import Field, SQLModel, create_engine, Session, select
import json

class Trace(SQLModel, table=True):
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    session_id: str = Field(index=True)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    query: str
    filter_criteria: Optional[str] = Field(default=None, description="JSON string of filters")
    generated_code: Optional[str] = Field(default=None)
    execution_result: Optional[str] = Field(default=None, description="JSON string of result")
    verification_score: Optional[float] = Field(default=None)
    final_answer: Optional[str] = Field(default=None)

class ChatSession(SQLModel, table=True):
    id: str = Field(primary_key=True)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata_json: Optional[str] = Field(default=None, description="JSON string of session metadata")

sqlite_file_name = "venra.db"
sqlite_url = f"sqlite:///{sqlite_file_name}"
engine = create_engine(sqlite_url, echo=False)

def init_db():
    SQLModel.metadata.create_all(engine)

def get_session():
    with Session(engine) as session:
        yield session

if __name__ == "__main__":
    init_db()
    print(f"Database {sqlite_file_name} initialized.")
