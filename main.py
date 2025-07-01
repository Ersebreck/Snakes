import os
from langchain_anthropic import ChatAnthropic
from utils import load_api_key, process_raw_interviews, load_codebook, save_results, build_workflow, load_coded_examples
import json

# Configure API key
os.environ["ANTHROPIC_API_KEY"] = load_api_key()

# Initialize the LLM
api_key = load_api_key()
anthropic_llm = ChatAnthropic(
    model="claude-3-sonnet-20240229",
    temperature=0,
    max_tokens=1000,
    anthropic_api_key=api_key
)


def main():
    # Configuración
    RAW_DATA_DIR = "data/raw_interviews"
    OUTPUT_DIR = "data/results/"
    CODEBOOK_PATH = "data/codebook/codebook.json"
    EXAMPLES_DIR = "data/coded_examples"
    
    # Archivos específicos a procesar
    specific_files = os.listdir(RAW_DATA_DIR) 
    specific_files = specific_files[:3]
    specific_files.append("interview_001_Interview 1 2-3-25-01 – Cuatro Esquinas_coded.txt")
    specific_files.append("interview_163_Interview 148 3-04-25-01 – Missiones_coded.txt")
    # Procesar entrevistas crudas
    print("\n=== Procesando entrevistas crudas ===")
    process_raw_interviews(
        raw_dir=RAW_DATA_DIR,
        output_dir=OUTPUT_DIR,
        specific_files=specific_files
    )
    
    # Load codebook
    print("Loading codebook...")
    try:
        codebook = load_codebook(CODEBOOK_PATH)
        if not isinstance(codebook, list):
            raise ValueError("Codebook should be a list of code dictionaries")
        print(f"Loaded codebook with {len(codebook)} codes")
    except Exception as e:
        print(f"Error loading codebook: {str(e)}")
        return
    
    # Load coded examples
    print("Loading coded examples...")
    coded_examples = load_coded_examples(EXAMPLES_DIR)
    print(f"Loaded {len(coded_examples)} coded examples")
    
    # Create workflow
    print("\nInitializing coding system...")
    
    # Process each interview
    results = []
    for interview_file in os.listdir(OUTPUT_DIR + "/to_code"):
        if interview_file.endswith("_coded.json"):
            interview_path = os.path.join(OUTPUT_DIR, "to_code", interview_file)
            with open(interview_path, 'r', encoding='utf-8') as f:
                coded_interview = json.load(f)
            
            interview_id = coded_interview["id"]
            print(f"\n📝 Processing interview: {interview_id}")
            
            # Initialize state for this interview
            state = {
                "anthropic_llm": anthropic_llm,
                "codebook": codebook,
                "interview_id": interview_id,
                "coded_examples": coded_examples,
                "interview_data": coded_interview  # Pass entire interview
            }
            
            # Execute workflow
            try:
                app = build_workflow()
                state = app(state)  # Use callable instead of invoke method
                
                # Collect results
                results.append({
                    "interview_id": interview_id,
                    "file_name": interview_file,
                    "assigned_codes": state.get("assigned_codes", []),
                    "analysis": state.get("analysis", ""),
                    "validation": state.get("validation", "")
                })
                
                print(f"  ✅ Processed interview {interview_id} with {len(state.get('assigned_codes', []))} codes")
            except Exception as e:
                error_msg = f"  ❌ Error processing interview: {str(e)}"
                print(error_msg)
                results.append({
                    "interview_id": interview_id,
                    "file_name": interview_file,
                    "error": str(e)
                })
    
    # Save results
    save_results(results)
    print("\nCoding process completed!")
    
if __name__ == "__main__":
    main()
