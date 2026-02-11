from pydantic import BaseModel, Field
from typing import Literal

class AuditVerdict(BaseModel):
    label: Literal["Supported", "Unfounded", "General"] = Field(
        ..., description="The classification of the target sentence relative to the evidence."
    )
    forensic_analysis: str = Field(
        ..., description="A detailed explanation of why the sentence is supported or unfounded. Reference specific numbers from the text vs the trace. Aim for 2-4 clear sentences."
    )
    detected_error_span: str = Field(
        ..., description="The exact value, date, or entity in the target_sentence that is incorrect. If the whole sentence is irrelevant, state 'entire_sentence'."
    )
    confidence: float = Field(..., ge=0, le=1)