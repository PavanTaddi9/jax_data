import os
import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError
from google.generativeai import GenerativeModel, configure


# Load environment variables
load_dotenv()

# --- Configuration ---
TEMP_API_KEY = os.getenv("GOOGLE_API_KEY", "AIzaSyBo2NcQbGXHum7ZcV3dA4ob9L4fnlQAPe0")
TEMPERATURE = 0.7

# --- Pydantic Models ---
class CodeExample(BaseModel):
    description: str = Field(..., description="Brief description of the code example")
    code: str = Field(..., description="Actual code content")

class InstructionQuery(BaseModel):
    query: str = Field(..., description="Natural language instruction for code generation")
    concepts: List[str] = Field(..., description="List of related concepts")

class JAXPromptInput(BaseModel):
    title: str
    concepts: List[str]
    code_examples: List[CodeExample]
class ProcessingStatus(BaseModel):
    item_id: int = Field(..., description="Input item ID")
    status: str = Field(..., description="Processing status")
    timestamp: str = Field(..., description="Processing time")
    error: Optional[str] = Field(None, description="Error message")

# --- Instruction Generator ---
class JAXInstructionGenerator:
    def __init__(self):
        if not TEMP_API_KEY or TEMP_API_KEY == "your-api-key-here":
            raise ValueError("API key not configured. Set GOOGLE_API_KEY in .env or TEMP_API_KEY variable")
        
        configure(api_key=TEMP_API_KEY)
        self.model = GenerativeModel("gemini-2.0-flash")
        self.generation_config = {
            "temperature": TEMPERATURE,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 8192,
        }
  
    
    def _create_instruction_prompt(self, data: JAXPromptInput) -> str:
        return f"""You are expert in generating high-quality, diverse natural language instructions for JAX code generation.

### Input
**Title**: {data.title}
**Concepts**:
{chr(10).join(f"- {c}" for c in data.concepts)}
**Code Examples**:
{chr(10).join(f"- {ce.description}:\\n{ce.code.replace('\\n', ' ')}" for ce in data.code_examples)}

### Task
Generate 10-15 diverse instructions for JAX coding tasks with type distribution:
1. Code Implementation Tasks (40-60%):
   - Direct code creation challenges
   - Algorithm implementations
   - Function/module development

2. Debugging Challenges (20-25%):
   - Identify/fix performance issues
   - Resolve API misuse errors
   - Fix parallelization bugs

3. Conceptual Explanations with Code (15-20%):
   - Explain JAX concepts with examples
   - Compare different approaches
   - Demonstrate paradigm differences

4. Performance Optimization (15-20%):
   - Vectorization tasks
   - JIT compilation improvements
   - Memory optimization challenges

5. API Usage Scenarios (5-10%):
   - Version-specific features
   - Hardware acceleration usage
   - Distributed computing patterns
```json
{{
  "instructions": [
    {{
      "query": "Create a JAX function that...",
      "concepts": ["jax.jit", "array operations"]
    }}
  ]
}}
```"""

    def generate_instructions(self, input_data: JAXPromptInput) -> List[InstructionQuery]:
        try:
            response = self.model.generate_content(
                contents=self._create_instruction_prompt(input_data),
                generation_config=self.generation_config
            )
            json_str = response.text.replace("```json", "").replace("```", "").strip()
            result = json.loads(json_str)
            return [InstructionQuery(**item) for item in result.get("instructions", [])]
        except Exception as e:
            print(f"Generation error: {str(e)}")
            return []

    def batch_generate(self, input_file: Path, output_file: Path):
        """Process JSON input with progress tracking"""
        all_instructions = []
        error_log = []
        status_log = []
        
        try:
            with open(input_file, "r") as f:
                input_data = json.load(f)
                
                for idx, item in enumerate(input_data, 1):
                    try:
                        validated = JAXPromptInput(**item)
                        instructions = self.generate_instructions(validated)
                        all_instructions.extend([inst.model_dump() for inst in instructions])
                        
                        status_log.append(ProcessingStatus(
                            item_id=idx,
                            status="success",
                            timestamp=datetime.now().isoformat(),
                            error=None
                        ).model_dump())
                        
                    except (ValidationError, json.JSONDecodeError) as e:
                        error_msg = f"Item {idx}: Validation error - {str(e)}"
                        error_log.append(error_msg)
                        status_log.append(ProcessingStatus(
                            item_id=idx,
                            status="error",
                            timestamp=datetime.now().isoformat(),
                            error=error_msg
                        ).model_dump())
                    except Exception as e:
                        error_msg = f"Item {idx}: Processing error - {str(e)}"
                        error_log.append(error_msg)
                        status_log.append(ProcessingStatus(
                            item_id=idx,
                            status="error",
                            timestamp=datetime.now().isoformat(),
                            error=error_msg
                        ).model_dump())
                        
        except json.JSONDecodeError as e:
            error_log.append(f"Invalid JSON file: {str(e)}")
        
        # Save main output
        with open(output_file, "w") as f:
            json.dump(all_instructions, f, indent=2)
            
        # Save processing status
        status_file = input_file.parent / f"{input_file.stem}_status.json"
        with open(status_file, "w") as f:
            json.dump(status_log, f, indent=2)
            
        # Save errors separately
        if error_log:
            error_path = output_file.parent / "generation_errors.log"
            with open(error_path, "w") as f:
                f.write("\n".join(error_log))

# --- Execution ---
if __name__ == "__main__":
    try:
        generator = JAXInstructionGenerator()
        
        input_path = Path("jax_doc.json")
        output_path = Path("jax_instructions.json")
        
        if input_path.exists():
            generator.batch_generate(input_path, output_path)
            print(f"Generated {output_path}")
            print(f"Status tracking: {input_path.stem}_status.json")
        else:
            print(f"Missing input file: {input_path}")
            
    except ValueError as e:
        print(f"Configuration Error: {str(e)}")
    except Exception as e:
        print(f"Unexpected Error: {str(e)}")