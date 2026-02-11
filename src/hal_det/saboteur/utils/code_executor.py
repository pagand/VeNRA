import math
from typing import Optional

def safe_execute_trace(code: str) -> Optional[str]:
    """
    Executes a Python trace string safely and captures the output.
    Uses a single execution block with proper locals/globals separation.
    """
    if not code or "# VERIFICATION" in code:
        return None

    safe_globals = {
        "__builtins__": {},
        "abs": abs, "round": round, "min": min, "max": max, "sum": sum, "len": len,
        "math": math
    }
    
    captured = {"result": None}
    
    def capture_print(*args):
        if args:
            captured["result"] = args[0]
            
    safe_globals["print"] = capture_print
    local_vars = {}
    
    try:
        # SINGLE EXECUTION
        exec(code, safe_globals, local_vars)
        
        # Priority 1: Explicit print capture
        if captured["result"] is not None:
            res = captured["result"]
        # Priority 2: Fallback to last assigned variable
        else:
            steps = [k for k in local_vars.keys() if k.startswith("step_")]
            if steps:
                # Sort numerically (step_1, step_2, step_10)
                steps.sort(key=lambda x: int(x.split('_')[1]))
                res = local_vars[steps[-1]]
            elif "result" in local_vars:
                res = local_vars["result"]
            else:
                return None

        # Format outcome
        if isinstance(res, (int, float)):
            if res == int(res):
                return str(int(res))
            return str(round(res, 2))
        return str(res)

    except Exception:
        # Fails safely on SyntaxError, ZeroDivisionError, etc.
        return None