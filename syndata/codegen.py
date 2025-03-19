import os
import json
import time
import hashlib
import traceback
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError
from google.generativeai import GenerativeModel, configure
import re
from tenacity import retry, stop_after_attempt, wait_exponential

# Load environment variables
load_dotenv()

# --- Configuration ---
TEMP_API_KEY = os.getenv("GOOGLE_API_KEY", "")
TEMPERATURE = 0.7

class TestCase(BaseModel):
    setup_code: str = Field(..., description="Code to set up the test environment")
    assertion: str = Field(..., description="Assertion to validate the solution")

class SolutionResponse(BaseModel):
    instruction: str = Field(..., description="Original instruction")
    solution_code: str = Field(..., description="Complete solution code")
    test_cases: List[TestCase] = Field(..., description="Validation test cases")

class ProcessingStatus(BaseModel):
    instruction_hash: str = Field(..., description="Hash of original instruction")
    status: str = Field(..., description="Processing status")
    timestamp: str = Field(..., description="Processing time")
    error: Optional[str] = Field(None, description="Error message")

# --- Solution Generator ---
class JAXSolutionGenerator:
    def __init__(self):
        if not TEMP_API_KEY or TEMP_API_KEY == "your-actual-api-key":
            raise ValueError("API key not configured. Set TEMP_API_KEY variable")
        
        configure(api_key=TEMP_API_KEY)
        self.model = GenerativeModel("gemini-2.0-flash")
        self.generation_config = {
            "temperature": TEMPERATURE,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 8192,
        }

    def _get_processed_hashes(self, done_file: Path) -> set:
        """Read processed instruction hashes from .done file"""
        try:
            return set(done_file.read_text().splitlines()) if done_file.exists() else set()
        except IOError as e:
            print(f"Error reading done file: {str(e)}")
            return set()

    def _create_solution_prompt(self, instruction: str) -> str:
        return f"""**JAX Code Generation Task**
1. **JAX-Native Implementation**:
   - Use pure functions with no side effects
   - Avoid in-place modifications (JAX arrays are immutable)
   - Prefer JAX control flow (jax.lax.cond/scan/while) over Python flow in JIT regions
   - Use jax.numpy instead of standard numpy

2. **Performance Optimization**:
   - Apply @jax.jit decorator to performance-critical functions
   - Use vmap for automatic vectorization
   - Consider pmap for multi-device parallelism where appropriate
   - Leverage XLA optimizations through JAX primitives

3. **Test Cases**:
   - MUST validate the generated solution code directly
   - Each test must contain:
     1. `setup_code`: Prepare inputs for the solution code
        - Initialize variables
        - Create test data
        - Call generated functions/classes
     2. `assertion`: Verify output matches expectations
        - Use assert statements
        - Compare with expected results
        - Handle edge cases
   - Test components MUST:
     - Reference actual function/class names from solution_code
     - Use realistic test values
     - Cover 2 scenarios:
       - Standard valid input
       - Edge case input

### Critical Requirements
1. Escape ALL special characters in code blocks:
   - Newlines → \\n
   - Tabs → \\t
   - Quotes → \\"
2. Maintain valid JSON structure
3. Use ONLY double quotes
4. Wrap ALL code blocks in <code> tags

### Response Format
```json
{{
    "solution_code": "<code>import jax\\n\\n# Properly escaped code<\\/code>",
    "test_cases": [
        {{
            "setup_code": "<code># Escaped setup code\\narr = jnp.array([1,2,3])<\\/code>",
            "assertion": "<code>assert result == expected<\\/code>"
        }}
    ]
}}```

### Input Instruction
{instruction}"""

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def generate_solution(self, instruction_data: dict) -> SolutionResponse:
        """Generate solution with enhanced JSON handling"""
        try:
            response = self.model.generate_content(
                contents=self._create_solution_prompt(instruction_data["query"]),
                generation_config=self.generation_config
            )
            
           

            if not response.text:
                raise RuntimeError("Empty response from model")

            # Process JSON response
            json_str = self._process_json_response(response.text)
            result = self._parse_json(json_str)
            
            return SolutionResponse(
                instruction=instruction_data["query"],
                solution_code=self._clean_code(result["solution_code"]),
                test_cases=[TestCase(
                    setup_code=self._clean_code(tc["setup_code"]),
                    assertion=self._clean_code(tc["assertion"])
                ) for tc in result["test_cases"]]
            )

        except Exception as e:
            raise RuntimeError(f"Generation failed: {str(e)}") from e

    def _process_json_response(self, raw_response: str) -> str:
        """Extract and sanitize JSON from model response"""
        # Remove problematic control characters
        sanitized = re.sub(r'[\x00-\x1F\x7F]', '', raw_response)
        
        # Find JSON using multiple patterns
        patterns = [
            r'```json\s*({.*?})\s*```',
            r'```\s*({.*?})\s*```',
            r'({.*})'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, sanitized, re.DOTALL)
            if match:
                json_str = match.group(1).strip()
                
                return self._sanitize_json(json_str)
        
        raise RuntimeError("No JSON structure found in response")

    def _sanitize_json(self, json_str: str) -> str:
        """Fix common JSON formatting issues"""
        # Remove trailing commas
        json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
        
        # Process code blocks to escape newlines and tabs only
        def escape_code_block(match):
            content = match.group(1)
            # Escape newlines and tabs only; leave quotes as is to avoid over-escaping
            content = content.replace("\n", "\\n").replace("\t", "\\t")
            return f"<code>{content}</code>"
        
        json_str = re.sub(r'<code>(.*?)</code>', escape_code_block, json_str, flags=re.DOTALL)
        
        # Balance braces if needed
        open_braces = json_str.count('{') - json_str.count('}')
        if open_braces > 0:
            json_str += '}' * open_braces
        
        return json_str

    def _parse_json(self, json_str: str) -> dict:
        """Safe JSON parsing with detailed error handling"""
        try:
            return json.loads(json_str, strict=False)
        except json.JSONDecodeError as e:
            error_context = json_str[max(0, e.pos-50):min(len(json_str), e.pos+50)]
            print(f"JSON Parse Error: {e.msg}\nContext: ...{error_context}...")
            raise RuntimeError(f"JSON parsing failed: {e.msg}") from e

    def _clean_code(self, code: str) -> str:
        """Clean code blocks from formatting"""
        return re.sub(r'</?code>|```', '', code).strip()

    def batch_process(self, input_file: Path, output_file: Path):
        """Process instructions with enhanced error handling"""
        temp_output = output_file.with_suffix(".tmp")
        done_file = input_file.parent / f"{input_file.stem}.done"
        status_log = []

        try:
            processed_hashes = self._get_processed_hashes(done_file)
            
            with open(input_file, "r") as f_in:
                instructions = json.load(f_in)
                
            valid_entries = []
            
            # Open done file in append mode
            with open(done_file, "a") as f_done:
                for idx, instr in enumerate(instructions):
                    instr_hash = "unknown"
                    try:
                        if not isinstance(instr, dict) or "query" not in instr:
                            raise ValueError("Invalid instruction format")
                            
                        instr_hash = hashlib.sha256(instr["query"].encode()).hexdigest()
                        
                        if instr_hash in processed_hashes:
                            print(f"Skipping processed instruction {idx+1}")
                            continue

                        solution = self.generate_solution(instr)
                        valid_entries.append(solution.model_dump())
                        f_done.write(f"{instr_hash}\n")
                        
                        status_log.append(ProcessingStatus(
                            instruction_hash=instr_hash,
                            status="success",
                            timestamp=datetime.now().isoformat()
                        ).model_dump())
                        
                        print(f"✅ Processed {idx+1}/{len(instructions)}")

                    except Exception as e:
                        error_msg = f"{type(e).__name__}: {str(e)}"
                        print(f"❌ Error processing item {idx+1}: {error_msg}")
                        status_log.append(ProcessingStatus(
                            instruction_hash=instr_hash,
                            status="error",
                            timestamp=datetime.now().isoformat(),
                            error=error_msg[:1000]
                        ).model_dump())

            # Write the valid entries to the temporary output file using a file handle
            with open(temp_output, "w") as f_out:
                json.dump(valid_entries, f_out, indent=2)
            temp_output.replace(output_file)
            print(f"Generated {output_file} with {len(valid_entries)} solutions")

        except Exception as e:
            print(f"🛑 Fatal error: {str(e)}")
            if temp_output.exists():
                temp_output.unlink()
            raise
        finally:
            status_file = input_file.parent / f"{input_file.stem}_status.json"
            status_file.write_text(json.dumps(status_log, indent=2))
            print(f"Status log written to {status_file}")

# --- Execution ---
if __name__ == "__main__":
    try:
        generator = JAXSolutionGenerator()
        input_path = Path("fil.json")
        output_path = Path("jax_solutions.json")
        
        if input_path.exists():
            print(f"Starting processing of {input_path}")
            generator.batch_process(input_path, output_path)
            print(f"Processing completed. Check {output_path} and *_status.json")
        else:
            print(f"Error: Input file not found: {input_path}")
            print("Sample input format:")
            print(json.dumps([{"query": "Create JAX function..."}], indent=2))
            
    except Exception as e:
        print(f"🚨 Critical failure: {str(e)}\n{traceback.format_exc()}")
