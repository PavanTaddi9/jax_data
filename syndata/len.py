import os
import json

def count_functions_in_json_folder(folder_path):
    total_functions = 0
    
    # Iterate through all files in the directory
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    # Assuming each JSON file contains a list of functions
                    num_functions = len(data)
                    total_functions += num_functions
                    print(f"{filename}: {num_functions} functions")
                    
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
    
    print(f"\nTotal functions across all files: {total_functions}")
    return total_functions

# Usage - replace with your folder path
folder_path = "/Users/pavankumartaddi/Desktop/syndata/"
count_functions_in_json_folder(folder_path)