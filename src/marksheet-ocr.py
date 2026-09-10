"""
Document Classification and Field Extraction using llama-cpp-python.
Computes average token-level confidence using log probabilities.
"""
from dotenv import load_dotenv
import os
load_dotenv()
import json
import math
import base64
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Qwen25VLChatHandler

PROJECT_ROOT_PATH=Path(str(os.getenv("PROJECT_ROOT_PATH")))
MODEL_ROOT_PATH=Path(str(os.getenv("MODEL_ROOT_PATH")))
MODEL_PATH = MODEL_ROOT_PATH / "Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf"
MMPROJ_PATH = MODEL_ROOT_PATH / "mmproj-qwen-2.5-vl-7B-Instruct-F16.gguf"
SAMPLE_IMAGE_PATH= PROJECT_ROOT_PATH / "input"

MAX_CONTEXT_WINDOW=4096

def encode_image_to_base64(image_path: Path) -> str:
    """Encodes a local image to base64 data URI format."""
    path = image_path
    if not path.is_file():
        raise FileNotFoundError(f"Image not found at path: {str(image_path)}")

    ext = path.suffix.lower().replace(".", "")
    mime_type = "image/png" if ext == "png" else "image/jpeg"

    with open(path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")

    return f"data:{mime_type};base64,{encoded_string}"

def calculate_field_confidences(logprobs_content: List[Any], parsed_json: Dict[str, Any]) -> float:
    """
    Parses sequence logprobs and top_logprobs content to calculate:
    1. Overall generation confidence (mean of exponentiated logprobs).
    2. Per-field geometric mean confidence by mapping character offsets to JSON values.

    Args:
        logprobs_content: Content list under response["choices"][0]["logprobs"]["content"]
        parsed_json: The decoded JSON output from the model.

    Returns:
        float: overall_confidence
    """
    if not logprobs_content:
        return 0.0, {} # type: ignore

    full_text = ""
    token_offsets = []
    all_probabilities = []

    # 1. Map tokens to character boundaries in the reconstructed text stream
    for item in logprobs_content:
        token_str = item.get("token", "") # type: ignore
        logprob = item.get("logprob") # type: ignore
        
        if logprob is not None:
            prob = math.exp(logprob)
            all_probabilities.append(prob)
        else:
            logprob = 0.0
            prob = 0.0

        start_idx = len(full_text)
        full_text += token_str
        end_idx = len(full_text)

        token_offsets.append({
            "start": start_idx,
            "end": end_idx,
            "logprob": logprob,
            "prob": prob
        })

    # Overall mean probability
    overall_confidence = (
        round(sum(all_probabilities) / len(all_probabilities), 4) 
        if all_probabilities else 0.0
    )
    return overall_confidence # type: ignore
def load_llm () -> Llama:
    chat_handler = Qwen25VLChatHandler(clip_model_path=str(MMPROJ_PATH))
    llm = Llama(
            model_path=str(MODEL_PATH),
            mmproj_path=str(MMPROJ_PATH),
            chat_handler=chat_handler,
            n_ctx=MAX_CONTEXT_WINDOW,
            n_gpu_layers=-1,
            logits_all=True,
            verbose=False
        )
    return llm
def process_document_with_confidence(loaded_llm:Llama, image_path: Path, confidence_threshold: float = 0.95, top_n: int = 3) -> Tuple[Dict[str, Any], bool]:
    """
    Executes vision inference, parses output JSON, and calculates model confidence.

    Args:
        image_path (Path): Path to document image.
        confidence_threshold (float): Minimum confidence acceptable without human review.

    Returns:
        Tuple[Dict[str, Any], bool]: Extracted data with metadata, and a flag indicating low confidence.
    """
    startTime = datetime.now()

    image_uri = encode_image_to_base64(image_path)
    with open("src/prompts/marksheet_ocr_sys.txt","r") as system_prompt_file:
        system_prompt=system_prompt_file.read()
        # print(system_prompt)

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_uri}},
                {"type": "text", "text": "Classify this document and extract, derive and fill text into the JSON format."}
            ]
        }
    ]

    # Request logprobs along with chat completion
    response = loaded_llm.create_chat_completion(
        messages=messages,
        temperature=0.0,
        max_tokens=1024,
        response_format={"type": "json_object"},
        logprobs=True,
        top_logprobs=top_n
    )

    choice = response["choices"][0] # type: ignore
    raw_json_str = choice["message"]["content"].strip() # type: ignore
    # logprobs_data = choice.get("logprobs", {}).get("content", [])

    try:
        parsed_data = json.loads(raw_json_str)
    except json.JSONDecodeError:
        parsed_data = {"error": "JSONDecodeError", "raw_output": raw_json_str}
        requires_review = True

    # Extract logprobs structure
    logprobs_obj = choice.get("logprobs")
    logprobs_content = logprobs_obj.get("content", []) if isinstance(logprobs_obj, dict) else []

    # Compute overall and field-level confidence via separate helper function
    overall_confidence_float = calculate_field_confidences(
        logprobs_content=logprobs_content,  # type: ignore
        parsed_json=parsed_data
    )

    usage = response.get("usage", {}) # type: ignore
    prompt_tokens = usage.get("prompt_tokens", 0)       # Input tokens (includes text + visual image tokens)
    completion_tokens = usage.get("completion_tokens", 0) # Output tokens generated
    total_tokens = usage.get("total_tokens", 0)           # Total context window used
    
    # Human-in-the-loop flag determination
    requires_review = (overall_confidence_float < confidence_threshold) or ("error" in parsed_data)
    endTime = datetime.now()
    elapsedSeconds= (endTime-startTime).total_seconds()
    output_payload = {
        "image_path": f"{image_path.parent.name}/{image_path.name}",
        "extracted_data": parsed_data,
        "token_usage": {
            "input_tokens": prompt_tokens,
            "output_tokens": completion_tokens,
            "total_context_used": total_tokens,
            "max_context_capacity": MAX_CONTEXT_WINDOW
        },
        "metrics": {
            "overall_confidence": overall_confidence_float,
            "requires_human_review": requires_review,
            "elapsed_seconds":elapsedSeconds
        },
    }

    return output_payload, requires_review


if __name__ == "__main__":
    # sample_img = SAMPLE_IMAGE_PATH
    # if os.path.exists(sample_img):
    #     llm=load_llm()
    #     payload, needs_flag = process_document_with_confidence(llm,sample_img)
    #     print(json.dumps(payload, indent=2, ensure_ascii=False))
    #     print(f"\nSent to Active Learning/RLHF Queue? {needs_flag}")
    json_list=[]
    llm=load_llm()
    input_image_count=0
    manual_review_count=0
    for image_file in SAMPLE_IMAGE_PATH.iterdir():
        if image_file.is_file():
           print("-" * 80)
           print(f"Input File Name:{image_file.parent.name}/{image_file.name}")
           input_image_count += 1
           payload, needs_flag = process_document_with_confidence(llm,image_file) 
           json_list.append(payload)
           if needs_flag == True:
            manual_review_count += 1     
           print(f"Manual Review Required: {needs_flag}")
           print("-" * 80)
    print(f"Input Images Processed:{ input_image_count }\nJson Array Output: { len(json_list) }\nManual review required count: {manual_review_count}")
    now = datetime.now()
    outjson = "output/"+now.strftime("%Y-%m-%d_%H%M%S")+".json"
    with open(outjson, "w") as file:
        json.dump(json_list, file, indent=4)  # indent=4 makes it pretty-printed