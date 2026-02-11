import re
import requests
from typing import List, Dict, Any
from venra.models import VerificationRequest, VerificationResponse, SentenceVerification
from venra.logging_config import logger

class SentinelJudge:
    """
    The Sentinel Judge is responsible for verifying the groundedness of an agent's response.
    It splits the response into sentences and validates each one against the source context and trace.
    """
    def __init__(self, api_url: str = "https://pagand-venra-haldet.hf.space/verify"):
        self.api_url = api_url

    def split_into_sentences(self, text: str) -> List[str]:
        """
        Splits text into sentences using a regex that respects common abbreviations.
        """
        # Regex: Splits after . ! or ? followed by whitespace, 
        # avoiding common abbreviations like Inc., Co., etc.
        sentence_endings = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s'
        sentences = re.split(sentence_endings, text)
        return [s.strip() for s in sentences if s.strip()]

    def verify_sentence(self, sentence: str, context: str, trace: str) -> Dict[str, Any]:
        """
        Calls the external SLM Judge for a single sentence.
        """
        payload = {
            "sentence": sentence,
            "context": context,
            "trace": trace
        }
        
        try:
            # Increased timeout to 60s to allow HF Space to wake up
            response = requests.post(self.api_url, json=payload, timeout=60)
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Judge API Error ({response.status_code}): {response.text}")
                return {"error": f"API Status {response.status_code}"}
        except requests.exceptions.Timeout:
            logger.error("Judge API timed out after 60s")
            return {"error": "Connection timed out (Space might be waking up)"}
        except Exception as e:
            logger.error(f"Failed to connect to Judge API: {e}")
            return {"error": str(e)}

    def verify_answer(self, request: VerificationRequest) -> VerificationResponse:
        """
        Processes the full answer: splits into sentences, verifies each, and aggregates results.
        """
        sentences = self.split_into_sentences(request.answer_text)
        results = []
        grounded_count = 0

        for sentence in sentences:
            logger.debug(f"Verifying sentence: {sentence[:50]}...")
            judge_res = self.verify_sentence(sentence, request.context, request.trace)
            
            if "error" in judge_res:
                # Fallback in case of API error
                results.append(SentenceVerification(
                    sentence=sentence,
                    label="HALLUCINATION", # Conservative fallback
                    grounded_prob=0.0,
                    common_prob=0.0,
                    hallucination_prob=1.0,
                    explanation=f"Error during verification: {judge_res['error']}"
                ))
                continue

            # Robust label determination based on the user's specific API output:
            # The API returns 'prediction' as the winner and lowercase keys for probabilities.
            label = judge_res.get("prediction", "HALLUCINATION")
            
            # Extract probabilities using the exact lowercase keys from the API
            grounded_p = judge_res.get("grounded", 0.0)
            common_p = judge_res.get("common", 0.0)
            hallucination_p = judge_res.get("hallucination", 0.0)

            verification = SentenceVerification(
                sentence=sentence,
                label=label,
                grounded_prob=grounded_p,
                common_prob=common_p,
                hallucination_prob=hallucination_p,
                explanation=judge_res.get("reasoning", None)
            )
            
            if verification.label == "GROUNDED":
                grounded_count += 1
            
            results.append(verification)

        overall_score = grounded_count / len(sentences) if sentences else 0.0
        
        return VerificationResponse(
            overall_groundedness_score=overall_score,
            sentence_results=results
        )
