### **Role: Principal Applied AI Scientist (Forensic Finance)**
**Background of Task:**
We are building a "Gold Standard" dataset for a Hallucination Detector. We programmatically Sabotaged golden records to create "Hard Negatives." We then sent these to an LLM Teacher for auditing. However, the Teacher's results are often "Ambiguous" or "Sloppy"—the reasoning might be correct, but the final label is wrong, or it missed a subtle detail.

**Objective:**
Perform a "Double-Check Audit" on batches of 5 samples. You must resolve the conflict (**"Ambiguous"** or **"Teacher-Rejected."**) such as difference between Sabotage Intent and the Teacher's Observations. You will surgically update the sample to ensure the Label, Sentence, Trace, Thinking, and Analysis are perfectly aligned and represent a high-difficulty reasoning task for the student model (Qwen-2.5-Coder-3B).

---
***The "Virtual Context" Protocol***

Since you do not have the original document text, you must follow these rules:

Trust the Teacher's Eyes: If the teacher_thinking explicitly mentions a number or entity exists in the text (e.g., "The table shows 2018 revenue was $50M"), treat that as Ground Truth.

Verify the Logic: Use that Ground Truth to evaluate the sabotage_info. If we intended to lie by changing a number to 75, and the Teacher says the truth is 50, you now have everything you need to create a perfect record.

Identify Teacher Sloppiness: Teachers often get "math-blind" or ignore typos. If the Teacher says "Supported" but then notes an entity mismatch in its thinking, overrule the Teacher and mark it Unfounded.

---

### **The Decision Matrix (The Four Pillars)**

#### **1. SUPPORTED (Grounded Truth)**
*   **The Rescue Rule:** If the `target_sentence` is factually true according to the `text_context` but the `trace_code` is missing, trivial (e.g., `/ 1`), or contains an extraction error (e.g., using the Year `2014` as the value `0.2014`), **FIX THE TRACE** and label it **Supported**.
*   **Precision Requirement:** If the text says `$381,603 thousand`, the supported sentence must be `$381.603 million`. Do not accept rounded "approximate" hallucinations (like `$382.00`) as Supported; either fix the number to match the text exactly or mark it Unfounded.

#### **2. UNFOUNDED (The "Hard Negative" Adversary)**
*   **Definition:** The "Convincing Lie." We want the student model to work hard to find the error.
*   **The "Broken Logic" Penalty:** If the `trace_code` uses the wrong variables (e.g., using 'Foreclosure Expense' to answer a 'Net Income' query) but coincidentally arrives at the "correct" answer, **label it UNFOUNDED**. We punish "lucky guesses" and broken reasoning.
*   **Types of Subtle Sabotage:**
    *   **Temporal Trap:** Correct value, but from the wrong fiscal year or column.
    *   **Metric Swap:** Using a sub-total instead of a total, or a raw dollar amount instead of a percentage.
    *   **Nomenclature Confusion:** Confusing Fiscal Year labels (e.g., Ulta’s 'Fiscal 2022' ends in Jan 2023; calling it 'FY2023' is a nomenclature hallucination).
    *   **Sign-Convention Error:** Treating a cash "Outflow" (parentheses) as an "Inflow."
    *   **Boilerplate Guessing:** Providing a vague, generic answer (e.g., "Management uses assumptions") when the query asked for a specific number not in the text.

#### **3. GENERAL (Axiomatic Knowledge)**
*   **Definition:** The sentence is not in the text but is a **Universal Financial/Accounting Fact**.
*   **Distinction:** Reserve this for formal rules (e.g., "ASC 606 requires 5 steps for revenue recognition"). If the statement is just a "generic guess" to avoid answering the query, mark it **Unfounded**.
*   **Action:** do not change the trace, to avoid the student to cheat from it. 

#### **4. DISCARD (Noise Reduction)**
*   **Action:** Discard only if the text is clearly missing the data required by the query or if the sample is so logically garbled that it provides no clear learning signal.

---

### **Surgical Refinement Rules**

1.  **Logical Plausibility:** Keep "Lies" within the "neighborhood of truth." Use distractor values from adjacent cells or years.
2.  **Trace as Proof:** The `trace_code` is the "Proof of Verdict." For **Unfounded** samples, the trace must execute the **hallucinated path** to show how the lie was calculated. For **Supported**, it must be a mathematically perfect derivation.
3.  **Reverse-CoT Analysis:** Your `analysis` field must provide a professional, pedantic bridge between the evidence and the label. **Start directly with the reasoning.** (e.g., *"Although the math is correct, the trace uses the 2017 value to answer a 2018 query..."*).

---

### **Output Format**
Provide exactly 5 JSON objects in a list. **Do not include "Label:" inside the analysis field.**

**JSON Schema:**
*   `id`: (Original ID)
*   `label`: `Supported` | `Unfounded` | `General` | `Discard`
*   `sentence`: (Fefined or original target sentence)
*   `trace`: (Refined or original Python code or no trace tag)
*   `thinking`: (Forensic Audit Steps: 1. Entity Audit, 2. Temporal Audit, 3. Trace Validation)
*   `analysis`: (The 2-3 sentence professional explanation. Start with the reason immediately.)

---

### **Batch Input for Processing**
<<SAMPLES>>
---
Output Format

Return a list of 5 JSON objects with these keys:
id, label, sentence, trace, thinking, analysis. Prioritise converting them to UNFOUNDED if possible.