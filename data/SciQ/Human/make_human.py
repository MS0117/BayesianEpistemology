import json

# Load the original questions and their supports from the provided files
original_file_path = '/userhomes/minsu/EviConf/data/SciQ/Final/200/SciQ_original_200.jsonl'
contradict_file_path = '/userhomes/minsu/EviConf/data/SciQ/Final/200/SciQ_contradict_50_200.jsonl'
incomplete_file_path = '/userhomes/minsu/EviConf/data/SciQ/Final/200/SciQ_incomplete_50_200.jsonl'
negate_file_path = '/userhomes/minsu/EviConf/data/SciQ/Final/200/SciQ_negate_100_200.jsonl'
irr_file_path = '/userhomes/minsu/EviConf/data/SciQ/Final/200/SciQ_irr_200.jsonl'
coincidence_file_path = '/userhomes/minsu/EviConf/data/SciQ/Final/200/SciQ_coincidence_remove_personal_200.jsonl'

# Reading all files into lists of dictionaries
def read_jsonl_file(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(json.loads(line))
    return data

original_data = read_jsonl_file(original_file_path)
contradict_data = read_jsonl_file(contradict_file_path)
incomplete_data = read_jsonl_file(incomplete_file_path)
negate_data = read_jsonl_file(negate_file_path)
irr_data = read_jsonl_file(irr_file_path)
coincidence_data = read_jsonl_file(coincidence_file_path)

# Updated indices for extraction (convert to zero-based index)
indices_to_extract = [
    2, 3, 4, 6, 7, 13, 14, 17, 19, 21, 26, 27, 40, 46, 50, 64, 66, 67, 92, 98,
    103, 113, 120, 121, 126, 127, 133, 137, 144, 152, 156, 174, 179, 180, 185,
    189, 190, 193, 194, 195, 199, 200
]
indices_to_extract = [i - 1 for i in indices_to_extract]  # Convert to zero-based

# Get the list of questions from the original data based on the given indices
questions_to_extract = [original_data[index]['question'] for index in indices_to_extract]


# Function to find the support from a specific dataset based on the question
def find_support(data, question):
    for entry in data:
        if entry['question'] == question:
            return entry['support']
    return None  # Return None if the question is not found


def find_coincidental_support(data, question):
    for entry in data:
        if entry['question'] == question:
            return entry['coincidental support']
    return None  # Return None if the question is not found


# Create the desired jsonl output
output_data = []
for question in questions_to_extract:
    entry = {
        'question': question,
        'answer': next((item['correct_answer'] for item in original_data if item['question'] == question), None),
        'original_support': find_support(original_data, question),
        'negated_support': find_support(negate_data, question),
        'coincidence_support': find_coincidental_support(coincidence_data, question),
        'irr_support': find_support(irr_data, question),
        'contradict_support': find_support(contradict_data, question),
        'incomplete_support': find_support(incomplete_data, question),


    }

    # Skip entry if any support is None
    if all(value is not None for value in entry.values()):
        output_data.append(entry)

# Save the output as jsonl file
output_file_path = '/userhomes/minsu/EviConf/data/SciQ/Final/Human/human_eval.jsonl'
with open(output_file_path, 'w') as outfile:
    for entry in output_data:
        json.dump(entry, outfile)
        outfile.write('\n')

output_file_path