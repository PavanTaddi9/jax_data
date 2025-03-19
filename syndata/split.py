import json
import re

input_file  = "jax_instructions.json"
outputfile = "fil.json"

with open(input_file,'r') as f:
  data = json.load(f)
  function_queries = [
        item for item in data 
        if "query" in item and 
        "function" in item["query"].lower()
    ]
    
    # Save filtered items to a new file
  with open(outputfile, 'w') as f:
      json.dump(function_queries, f, indent=2)
    
  print(f"Found {len(function_queries)} items containing 'function' in their query")
  print(f"Results saved to {outputfile}")