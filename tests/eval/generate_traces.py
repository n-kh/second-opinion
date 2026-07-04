import os
import json
import base64
import asyncio
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.artifacts import InMemoryArtifactService
from google.adk.memory import InMemoryMemoryService
from google.adk.events.request_input import RequestInput
from google.genai import types

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.agent import app as adk_app

DATASET_PATH = "tests/eval/datasets/basic-dataset.json"
OUTPUT_PATH = "artifacts/traces/generated_traces.json"

def serialize_part(part) -> dict:
    d = {}
    if part.text:
        d["text"] = part.text
    if part.function_call:
        fc = part.function_call
        d["function_call"] = {
            "name": fc.name,
            "args": fc.args,
            "id": fc.id
        }
    if part.function_response:
        fr = part.function_response
        d["function_response"] = {
            "name": fr.name,
            "response": fr.response,
            "id": fr.id
        }
    return d

def serialize_content(content) -> dict:
    if not content:
        return {"role": "model", "parts": []}
    parts = []
    for part in content.parts:
        parts.append(serialize_part(part))
    return {
        "role": content.role or "model",
        "parts": parts
    }

async def run_scenario(case_id: str, prompt_text: str) -> dict:
    session_service = InMemorySessionService()
    artifact_service = InMemoryArtifactService()
    memory_service = InMemoryMemoryService()
    
    runner = Runner(
        app=adk_app,
        session_service=session_service,
        artifact_service=artifact_service,
        memory_service=memory_service,
        auto_create_session=True,
    )
    
    user_id = "eval_user"
    session_id = f"eval_{case_id}"
    
    # Turn 0 input
    turn0_events = []
    user_part = types.Part.from_text(text=prompt_text)
    user_content = types.Content(role="user", parts=[user_part])
    
    turn0_events.append({
        "author": "user",
        "content": serialize_content(user_content)
    })
    
    first_run_events = []
    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=user_content
    ):
        first_run_events.append(event)
        
    for event in first_run_events:
        if event.content:
            turn0_events.append({
                "author": "medical_assistant",
                "content": serialize_content(event.content)
            })
            
    turns = [
        {
            "turn_index": 0,
            "events": turn0_events
        }
    ]
    
    return {
        "eval_case_id": case_id,
        "prompt": serialize_content(user_content),
        "agent_data": {
            "agents": {
                "medical_assistant": {
                    "agent_id": "medical_assistant",
                    "instruction": "Medical history analysis and NCCN guidelines compliance checker."
                }
            },
            "turns": turns
        }
    }

async def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    
    with open(DATASET_PATH, "r") as f:
        dataset = json.load(f)
        
    eval_cases = []
    for case in dataset["eval_cases"]:
        case_id = case["eval_case_id"]
        prompt_text = case["prompt"]["parts"][0]["text"]
        print(f"Running scenario: {case_id}...")
        result_case = await run_scenario(case_id, prompt_text)
        eval_cases.append(result_case)
        
    output_data = {"eval_cases": eval_cases}
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"Traces successfully generated and written to {OUTPUT_PATH}")

if __name__ == "__main__":
    asyncio.run(main())
