import re
import os
from pathlib import Path
from typing import Dict, List, Optional
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END, add_messages
import json
from pathlib import Path
from typing import Dict, List, Any, TypedDict, Optional, Annotated
from datetime import datetime


# Load API key
def load_api_key():
    try:
        with open('ANTROPIC_API_KEY', 'r') as f:
            return f.read().strip()
    except FileNotFoundError:
        print("Error: ANTROPIC_API_KEY file not found")
        exit(1)

class State(TypedDict):
    messages: List[Dict[str, str]]  # List of message dictionaries with 'role' and 'content' keys
    assigned_codes: List[str]  # Now for entire interview
    codebook: List[Dict[str, Any]]
    interview_id: str
    analysis: Optional[str]  # Analysis of entire interview
    validation: Optional[str]  # Validation of entire interview
    anthropic_llm: Any  # Pass the LLM instance via state
    coded_examples: List[Dict]
    interview_data: Dict  # Entire interview data

def load_codebook(codebook_path: str) -> Dict[str, Any]:
    with open(codebook_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_coded_examples(examples_dir: str) -> List[Dict]:
    """
    Load pre-coded example interviews from JSON files
    
    Args:
        examples_dir: Directory containing coded example files
    
    Returns:
        List of example interview objects with their coding annotations
    """
    examples = []
    example_files = Path(examples_dir).glob("*.json")
    
    for file_path in example_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                example_data = json.load(f)
                examples.append(example_data)
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON file: {file_path}")
    return examples

def parse_raw_interview(file_path: Path) -> Dict:
    """
    Parsea un archivo de texto de entrevista cruda en un formato estructurado.
    
    Args:
        file_path: Ruta al archivo de texto de entrevista cruda
    
    Returns:
        Un diccionario con claves:
          - title: el título de la entrevista (primera línea)
          - utterances: lista de diccionarios de expresiones
    """
    with open(str(file_path).replace("data/raw_interviews/data/raw_interviews/", "data/raw_interviews/"), 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # La primera línea es el título
    title = lines[0].strip()
    
    utterances = []
    utterance_id = 1
    index = 1  # Comenzar desde la segunda línea
    while index < len(lines):
        # Verificar si la línea tiene timestamp y hablante
        if lines[index].startswith('['):
            # Extraer timestamp y hablante
            parts = lines[index].split(' - ', 1)  # Dividir en dos partes en el primer ' - '
            if len(parts) < 2:
                index += 1
                continue
            timestamp_str = parts[0].strip()
            # Eliminar corchetes
            timestamp = timestamp_str[1:-1] if timestamp_str.startswith('[') and timestamp_str.endswith(']') else timestamp_str
            speaker = parts[1].strip()
            
            # La siguiente línea es el texto
            index += 1
            if index >= len(lines):
                break
            text = lines[index].strip()
            
            utterances.append({
                'utterance_id': str(utterance_id),
                'timestamp': timestamp,
                'speaker': speaker,
                'text': text,
                'codes': []
            })
            utterance_id += 1
        index += 1
    
    return {
        'title': title,
        'utterances': utterances
    }

def process_raw_interviews(raw_dir: str, output_dir: str, specific_files: List[str] = None):
    """
    Procesa entrevistas crudas y genera archivos codificados en formato JSON
    
    Args:
        raw_dir: Directorio con entrevistas crudas (.txt)
        output_dir: Directorio para guardar entrevistas codificadas (.json)
        specific_files: Lista opcional de archivos específicos a procesar
    """
    # Crear directorio de salida si no existe
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Obtener lista de archivos a procesar
    if specific_files:
        raw_files = [Path(raw_dir) / f for f in specific_files]
    else:
        raw_files = list(Path(raw_dir).glob("*.txt"))
    
    for file_path in raw_files:
        #try:
            print(f"Procesando: {file_path.name}")
            
            # Parsear entrevista cruda
            interview = parse_raw_interview(file_path)
            
            # Crear estructura para archivo codificado
            coded_interview = {
                "id": file_path.stem,
                "file_name": file_path.name,
                "title": interview["title"],
                "messages": []
            }
            
            # Convertir utterances a mensajes
            for utterance in interview["utterances"]:
                coded_interview["messages"].append({
                    "utterance_id": utterance["utterance_id"],
                    "timestamp": utterance["timestamp"],
                    "speaker": utterance["speaker"],
                    "text": utterance["text"],
                    "codes": utterance["codes"]
                })
            
            # Guardar como JSON
            output_path = Path(output_dir)/"to_code"/ f"{file_path.stem}_coded.json"
            with open(str(output_path).replace("/data/raw_interviews/", ""), 'w', encoding='utf-8') as f:
                json.dump(coded_interview, f, ensure_ascii=False, indent=4)
                
            print(f"✅ Entrevista codificada guardada en: {output_path}")
            
        #except Exception as e:
        #    print(f"❌ Error procesando {file_path.name}: {str(e)}")


def analyze_interview(state: State) -> State:
    interview = state["interview_data"]
    anthropic_llm = state["anthropic_llm"]
    
    # Extract all text from the interview
    full_text = "\n".join([f"[Speaker: {msg.get('speaker', 'Unknown')}, Timestamp: {msg.get('timestamp', 'N/A')}]\n{msg.get('text', '')}" for msg in interview["messages"]])
    
    prompt = f"""
    As a qualitative research analyst, analyze this entire interview and identify:
    1. Key themes and concepts
    2. Notable patterns across the conversation
    3. Overall speaker perspectives
    
    Provide a concise summary (3-5 sentences) focusing on the main insights.
    
    Interview Title: {interview.get('title', 'Untitled')}
    
    Full Interview Text:
    {full_text}
    """
    
    messages = [
        SystemMessage(content="You are a helpful assistant that analyzes interviews."),
        HumanMessage(content=prompt)
    ]
    response = anthropic_llm.invoke(messages)
    analysis = response.content if hasattr(response, 'content') else str(response)
    
    state["analysis"] = analysis
    
    if "messages" not in state:
        state["messages"] = []
        
    state["messages"].append({
        "role": "system",
        "content": f"Interview analysis: {analysis}"
    })
    
    return state

def assign_codes_interview(state: State) -> State:
    interview = state["interview_data"]
    anthropic_llm = state["anthropic_llm"]
    
    # Extract all text from the interview
    full_text = "\n".join([f"[Speaker: {msg.get('speaker', 'Unknown')}, Timestamp: {msg.get('timestamp', 'N/A')}]\n{msg.get('text', '')}" for msg in interview["messages"]])
    
    codebook = state.get("codebook", [])
    codebook_str = "\n".join([f"- {code.get('code_id', '')}: {code.get('description', '')}" for code in codebook])
    
    # Format examples for prompt
    example_str = ""
    for i, example in enumerate(state.get("coded_examples", [])[:3]):
        if "messages" in example:
            # For each example, show the entire interview text and the assigned codes
            example_text = "\n".join([f"[Speaker: {msg.get('speaker', 'Unknown')}, Timestamp: {msg.get('timestamp', 'N/A')}]\n{msg.get('text', '')}" for msg in example["messages"]])
            example_str += f"\nExample {i+1}:\n"
            example_str += f"Interview: {example.get('title', 'Untitled')}\n"
            example_str += f"Codes: {', '.join(example.get('codes', []))}\n"
    
    prompt = f"""
    As a qualitative coding expert, assign the most relevant codes from the codebook to this entire interview. 
    Focus on identifying overarching themes and patterns. Respond ONLY with comma-separated code IDs.
    
    Consider these coding examples:
    {example_str}
    
    Codebook:
    {codebook_str}
    
    ---
    
    Interview Title: {interview.get('title', 'Untitled')}
    
    Full Interview Text:
    {full_text}
    """
    
    messages = [
        SystemMessage(content="You are a helpful assistant that assigns codes to interviews. Respond ONLY with comma-separated codes from the provided codebook."),
        HumanMessage(content=prompt)
    ]
    response = anthropic_llm.invoke(messages)
    response_text = response.content if hasattr(response, 'content') else str(response)
    
    codes = [code.strip() for code in response_text.split(",") if code.strip()]
    state["assigned_codes"] = codes
    
    if "messages" not in state:
        state["messages"] = []
        
    state["messages"].append({
        "role": "system",
        "content": f"Assigned codes for interview: {', '.join(codes) if codes else 'No codes assigned'}"
    })
    
    return state

def validate_codes_interview(state: State) -> State:
    interview = state["interview_data"]
    anthropic_llm = state["anthropic_llm"]
    
    # Extract all text from the interview
    full_text = "\n".join([f"[Speaker: {msg.get('speaker', 'Unknown')}, Timestamp: {msg.get('timestamp', 'N/A')}]\n{msg.get('text', '')}" for msg in interview["messages"]])
    
    # Format examples for prompt
    example_str = ""
    for i, example in enumerate(state.get("coded_examples", [])[:2]):
        if "messages" in example:
            example_text = "\n".join([f"[Speaker: {msg.get('speaker', 'Unknown')}, Timestamp: {msg.get('timestamp', 'N/A')}]\n{msg.get('text', '')}" for msg in example["messages"]])
            example_str += f"\nExample {i+1}:\n"
            example_str += f"Interview: {example.get('title', 'Untitled')}\n"
            example_str += f"Codes: {', '.join(example.get('codes', []))}\n"
    
    prompt = f"""
    Review the assigned codes for this entire interview and validate their appropriateness. 
    For each code, provide a brief justification (1 sentence) why it is appropriate or not. 
    Suggest alternative codes if needed.
    
    Consider these validation examples:
    {example_str}
    
    Interview Title: {interview.get('title', 'Untitled')}
    
    Full Interview Text:
    {full_text}
    
    Assigned codes: {', '.join(state.get("assigned_codes", [])) if state.get("assigned_codes", []) else 'None'}
    """
    
    messages = [
        SystemMessage(content="You are a helpful assistant that validates code assignments for interviews."),
        HumanMessage(content=prompt)
    ]
    response = anthropic_llm.invoke(messages)
    validation = response.content if hasattr(response, 'content') else str(response)
    
    state["validation"] = validation
    
    if "messages" not in state:
        state["messages"] = []
        
    state["messages"].append({
        "role": "system",
        "content": f"Code validation for interview: {validation}"
    })
    
    return state

def save_results(results: List[Dict], output_dir: str = "data/results"):
    """Save the coding results to a JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save each interview result separately
    for interview in results:
        interview_id = interview['interview_id']
        interview_file = f"{output_dir}/{interview_id}_coding_results_{timestamp}.json"
        
        with open(interview_file.replace("/data/raw_interviews/", ""), 'w', encoding='utf-8') as f:
            json.dump(interview, f, ensure_ascii=False, indent=4)
    
    # Also save the combined results
    combined_file = f"{output_dir}/combined_coding_results_{timestamp}.json"
    with open(combined_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    
    print(f"✅ Results saved to: {output_dir}/*_coding_results_{timestamp}.json")
    print(f"✅ Combined results saved to: {combined_file}")

def build_workflow():
    workflow = StateGraph(State)
    
    workflow.add_node("analyze_interview", analyze_interview)
    workflow.add_node("assign_codes_interview", assign_codes_interview)
    workflow.add_node("validate_codes_interview", validate_codes_interview)
    
    workflow.set_entry_point("analyze_interview")
    workflow.add_edge("analyze_interview", "assign_codes_interview")
    workflow.add_edge("assign_codes_interview", "validate_codes_interview")
    workflow.add_edge("validate_codes_interview", END)
    
    app = workflow.compile()
    
    def wrapped_app(state):
        try:
            if "messages" not in state:
                state["messages"] = []
                
            if "assigned_codes" not in state:
                state["assigned_codes"] = []
                
            result = app.invoke(state)
            return result
            
        except Exception as e:
            error_msg = f"Workflow error: {str(e)}"
            print(error_msg)
            
            if "messages" not in state:
                state["messages"] = []
                
            state["messages"].append({
                "role": "system",
                "content": f"Error in workflow: {str(e)}"
            })
            
            if "assigned_codes" not in state:
                state["assigned_codes"] = []
                
            return state
    
    return wrapped_app
