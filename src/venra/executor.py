import sys
import io
import traceback
from typing import Dict, Any, Optional

class PythonExecutor:
    """
    Executes Python code in a controlled local environment.
    Captures stdout and returns the result or error.
    """
    def execute(self, code: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        context = context or {}
        
        # Redirect stdout
        old_stdout = sys.stdout
        redirected_output = io.StringIO()
        sys.stdout = redirected_output
        
        error = None
        try:
            # We use exec but provide a restricted global scope
            exec(code, {"__builtins__": __builtins__}, context)
        except Exception:
            error = traceback.format_exc()
        finally:
            sys.stdout = old_stdout
            
        output = redirected_output.getvalue()
        
        return {
            "output": output.strip(),
            "variables": {k: v for k, v in context.items() if not k.startswith("__")},
            "error": error
        }
