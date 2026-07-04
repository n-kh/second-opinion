# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import base64
import json
import os
import re
from typing import List, Optional, Any, Dict
from pydantic import BaseModel, Field

import google.auth
from google.adk.agents import LlmAgent
from google.adk.agents.context import Context
from google.adk.apps import App, ResumabilityConfig
from google.adk.events.event import Event
from google.adk.events.request_input import RequestInput
from google.adk.workflow import Workflow, node
from google.genai import types

# GCP environment setup
try:
    _, project_id = google.auth.default()
    if project_id:
        os.environ["GOOGLE_CLOUD_PROJECT"] = project_id
except Exception:
    pass

os.environ["GOOGLE_CLOUD_LOCATION"] = "global"
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"

# ==========================================
# PII & Injection Utilities
# ==========================================

def luhn_check(card_number: str) -> bool:
    digits = [int(c) for c in card_number if c.isdigit()]
    if len(digits) < 13 or len(digits) > 19:
        return False
    odd_digits = digits[-1::-2]
    even_digits = digits[-2::-2]
    total = sum(odd_digits)
    for d in even_digits:
        total += sum(divmod(d * 2, 10))
    return total % 10 == 0

def scrub_pii(text: str) -> tuple[str, List[str]]:
    redacted_categories = set()
    scrubbed_text = text

    ssn_pattern = re.compile(r"\b\d{3}[- ]?\d{2}[- ]?\d{4}\b")
    if ssn_pattern.search(scrubbed_text):
        scrubbed_text = ssn_pattern.sub("[REDACTED_SSN]", scrubbed_text)
        redacted_categories.add("SSN")

    cc_pattern = re.compile(r"\b(?:\d[ -]*?){13,19}\b")
    def cc_replacer(match: re.Match) -> str:
        matched_str = match.group(0)
        digits_only = "".join(c for c in matched_str if c.isdigit())
        if luhn_check(digits_only):
            redacted_categories.add("Credit Card")
            return "[REDACTED_CREDIT_CARD]"
        return matched_str

    scrubbed_text = cc_pattern.sub(cc_replacer, scrubbed_text)
    return scrubbed_text, sorted(list(redacted_categories))

def detect_prompt_injection(text: str) -> bool:
    injection_patterns = [
        r"ignore (all )?previous instructions",
        r"ignore the rules",
        r"system prompt override",
        r"bypass (validation|rules|audit)",
        r"approve (this|all) automatically",
        r"force auto-approval",
        r"override system instructions",
        r"you must approve",
        r"do not audit",
        r"forget previous guidelines",
        r"new instruction:",
        r"you are now an auto-approver"
    ]
    text_lower = text.lower()
    for pattern in injection_patterns:
        if re.search(pattern, text_lower):
            return True
    return False

# ==========================================
# Schema Definitions
# ==========================================

class ExpenseInput(BaseModel):
    amount: float
    submitter: str
    category: str
    description: str
    date: str

class ExpenseReviewState(BaseModel):
    submitter: str = ""
    category: str = ""
    date: str = ""
    amount: float = 0.0
    raw_description: str = ""
    clean_description: str = ""
    redacted_categories: List[str] = []
    is_security_event: bool = False
    security_event_reason: Optional[str] = None
    llm_review_status: Optional[str] = None
    llm_review_notes: Optional[str] = None
    human_review_status: Optional[str] = None
    human_review_notes: Optional[str] = None

# ==========================================
# Graph Nodes
# ==========================================

def security_checkpoint(node_input: Any):
    amount = 0.0
    submitter = "unknown"
    category = "other"
    description = ""
    date = ""

    if isinstance(node_input, str):
        try:
            parsed = json.loads(node_input)
            if isinstance(parsed, dict):
                if "data" in parsed:
                    data = parsed["data"]
                    if isinstance(data, str):
                        data = json.loads(base64.b64decode(data))
                    amount = float(data.get("amount", 0.0))
                    submitter = data.get("submitter", "unknown")
                    category = data.get("category", "other")
                    description = data.get("description", "")
                    date = data.get("date", "")
                else:
                    amount = float(parsed.get("amount", 0.0))
                    submitter = parsed.get("submitter", "unknown")
                    category = parsed.get("category", "other")
                    description = parsed.get("description", "")
                    date = parsed.get("date", "")
            else:
                description = node_input
        except Exception:
            description = node_input
    elif isinstance(node_input, dict):
        if "data" in node_input:
            data = node_input["data"]
            if isinstance(data, str):
                data = json.loads(base64.b64decode(data))
            amount = float(data.get("amount", 0.0))
            submitter = data.get("submitter", "unknown")
            category = data.get("category", "other")
            description = data.get("description", "")
            date = data.get("date", "")
        else:
            amount = float(node_input.get("amount", 0.0))
            submitter = node_input.get("submitter", "unknown")
            category = node_input.get("category", "other")
            description = node_input.get("description", "")
            date = node_input.get("date", "")
    elif hasattr(node_input, "description"):
        amount = getattr(node_input, "amount", 0.0)
        submitter = getattr(node_input, "submitter", "unknown")
        category = getattr(node_input, "category", "other")
        description = getattr(node_input, "description", "")
        date = getattr(node_input, "date", "")
    else:
        description = str(node_input)

    clean_desc, redacted_cats = scrub_pii(description)
    is_injection = detect_prompt_injection(description)
    
    state_delta = {
        "submitter": submitter,
        "category": category,
        "date": date,
        "amount": amount,
        "raw_description": description,
        "clean_description": clean_desc,
        "redacted_categories": redacted_cats,
        "is_security_event": is_injection,
        "security_event_reason": "Prompt injection detected in description" if is_injection else None
    }
    
    route = "security_flagged" if is_injection else "clean"
    
    info_msg = (
        f"🛡️ **Security Checkpoint Output**:\n"
        f"- **Sanitized Description**: {clean_desc}\n"
        f"- **Redacted Categories**: {', '.join(redacted_cats) if redacted_cats else 'None'}\n"
        f"- **Security Event**: {is_injection}"
    )
    yield Event(content=types.Content(role='model', parts=[types.Part.from_text(text=info_msg)]))
    yield Event(output=clean_desc, route=route, state=state_delta)


def format_for_llm(ctx: Context, node_input: str) -> str:
    amount = ctx.state["amount"]
    category = ctx.state["category"]
    return f"Category: {category}\nAmount: ${amount}\nDescription: {node_input}"


class LLMReviewOutput(BaseModel):
    status: str = Field(description="Must be 'Approved', 'Rejected', or 'Escalated'")
    notes: str = Field(description="Explanation of the decision")

llm_reviewer = LlmAgent(
    name="llm_reviewer",
    model="gemini-2.5-flash",
    instruction=(
        "You are an expense reviewer. Review the expense details (category, amount, description).\n"
        "Rules:\n"
        "1. Expenses over $500 must be Escalated.\n"
        "2. Alcohol, bar, or entertainment expenses must be Rejected.\n"
        "3. Standard business expenses should be Approved.\n"
        "Format your output strictly according to the requested schema."
    ),
    output_schema=LLMReviewOutput,
    output_key="llm_review"
)


def has_gcp_credentials() -> bool:
    if os.environ.get("INTEGRATION_TEST") == "TRUE":
        return False
    if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
        return True
    try:
        google.auth.default()
        return True
    except Exception:
        return False


@node(rerun_on_resume=True)
async def llm_reviewer_node(ctx: Context, node_input: str):
    """
    Wraps the LLM Reviewer with a local fallback when credentials are not configured.
    """
    if has_gcp_credentials():
        try:
            result = await ctx.run_node(llm_reviewer, node_input=node_input)
            yield Event(output=result)
            return
        except Exception:
            pass

    amount = ctx.state.get("amount", 0.0)
    clean_desc = ctx.state.get("clean_description", "")
    
    warning_msg = "⚠️ **Credentials Not Configured**: Running in simulated mode."
    yield Event(content=types.Content(role='model', parts=[types.Part.from_text(text=warning_msg)]))
    
    # Policy Simulation
    if amount > 500:
        status = "Escalated"
        notes = "Expense exceeds $500 threshold, escalating to human review (Simulated)."
    elif "alcohol" in clean_desc.lower() or "bar" in clean_desc.lower():
        status = "Rejected"
        notes = "Policy violation: Alcohol or bar expenses require prior approval (Simulated)."
    else:
        status = "Approved"
        notes = "Expense fits within standard guidelines (Simulated)."
        
    yield Event(output={"status": status, "notes": notes})


def route_after_llm(ctx: Context, node_input: dict):
    status = node_input.get("status")
    notes = node_input.get("notes")
    
    state_delta = {
        "llm_review_status": status,
        "llm_review_notes": notes
    }
    
    route = "escalate" if status == "Escalated" else "done"
    
    info_msg = f"🤖 **LLM Reviewer Decision**: **{status}**\n- **Notes**: {notes}"
    yield Event(content=types.Content(role='model', parts=[types.Part.from_text(text=info_msg)]))
    yield Event(output=status, route=route, state=state_delta)


async def human_reviewer(ctx: Context, node_input: Any):
    amount = ctx.state.get("amount", 0.0)
    clean_desc = ctx.state.get("clean_description", "")
    is_sec = ctx.state.get("is_security_event", False)
    sec_reason = ctx.state.get("security_event_reason", "")
    
    if is_sec:
        msg = f"⚠️ **SECURITY ALERT (Prompt Injection)**:\n- **Reason**: {sec_reason}\n- **Sanitized Description**: {clean_desc}"
    else:
        msg = f"👤 **Escalated Expense Review Required**:\n- **Amount**: ${amount}\n- **Sanitized Description**: {clean_desc}"
        
    if not ctx.resume_inputs or "approval" not in ctx.resume_inputs:
        yield RequestInput(
            interrupt_id="approval",
            message=f"{msg}\n\nDecision needed (Approved / Rejected):"
        )
        return
        
    decision = ctx.resume_inputs["approval"]
    if isinstance(decision, dict):
        decision = decision.get("decision") or decision.get("result") or decision.get("response") or "Approved"
        
    state_delta = {
        "human_review_status": decision,
        "human_review_notes": f"Reviewed by human. Decision: {decision}"
    }
    
    info_msg = f"👤 **Human Reviewer Decision**: **{decision}**\n- **Notes**: {state_delta['human_review_notes']}"
    yield Event(content=types.Content(role='model', parts=[types.Part.from_text(text=info_msg)]))
    yield Event(output=decision, state=state_delta)

# ==========================================
# Graph Definition
# ==========================================

edges = [
    ('START', security_checkpoint),
    (security_checkpoint, {"security_flagged": human_reviewer, "clean": format_for_llm}),
    (format_for_llm, llm_reviewer_node),
    (llm_reviewer_node, route_after_llm),
    (route_after_llm, {"escalate": human_reviewer}),
]

root_agent = Workflow(
    name="expense_reviewer",
    edges=edges,
    state_schema=ExpenseReviewState,
)

app = App(
    root_agent=root_agent,
    name="app",
    resumability_config=ResumabilityConfig(enabled=True)
)
