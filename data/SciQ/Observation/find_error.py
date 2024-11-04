import json

# Replace 'input_file1.json' and 'input_file2.json' with the actual file names
input_file1 = '/userhomes/minsu/EviConf/data/SciQ/Final/Observation/SciQ_observation_experiment.jsonl'
input_file2 = '/userhomes/minsu/EviConf/data/SciQ/Final/Observation/SciQ_observation_observation.jsonl'

with open(input_file1, 'r') as infile1, open(input_file2, 'r') as infile2:
    # Use zip to iterate over both files simultaneously
    for i, (line1, line2) in enumerate(zip(infile1, infile2)):
        try:
            # Parse the JSON object from each line
            data1 = json.loads(line1)
            data2 = json.loads(line2)

            # Compare the "question" key in both data dictionaries
            if data1['question'] != data2['question']:
                print(f"First differing question found at index {i+1}:")
                print(f"File 1 question: {data1['question']}")
                print(f"File 2 question: {data2['question']}")
                break
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON at line {i}: {e}")

    else:
        # If the loop completes without finding any differences
        print("No differing questions found in the lines compared.")