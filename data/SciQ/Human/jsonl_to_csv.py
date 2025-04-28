import pandas as pd

# Load the jsonl file
file_path = '/userhomes/minsu/EviConf/data/SciQ/Final/Human/human_eval.jsonl'

# Load JSONL into a DataFrame
data = pd.read_json(file_path, lines=True)

# Display the data structure to understand its columns
data.head()

expanded_data = []

# 1-25: Question only, no support
for idx, row in data.iterrows():
    question = row['question']

    expanded_data.append({
        'Question': question,
        'Description': "Provide your best guess and the probability that it is correct (1 to 10) for the following question.",
        'Support': '',
        'Answer': '',
        'Confidence': ''
    })

# 26-50: Original support included
for idx, row in data.iterrows():
    question = row['question']

    expanded_data.append({
        'Question': question,
        'Description': "Provide your best guess and the probability that it is correct (1 to 10) for the following question based on the support.",
        'Support': row['original_support'],
        'Answer': '',
        'Confidence': ''
    })

# 51-75: Negated support
for idx, row in data.iterrows():
    question = row['question']

    expanded_data.append({
        'Question': question,
        'Description': "Provide your best guess and the probability that it is correct (1 to 10) for the following question based on the support.",

        'Support': row['negated_support'],
        'Answer': '',
        'Confidence': ''
    })

# 76-100: Coincidence support
for idx, row in data.iterrows():
    question = row['question']

    expanded_data.append({
        'Question': question,
        'Description': "Provide your best guess and the probability that it is correct (1 to 10) for the following question based on the support.",

        'Support': row['coincidence_support'],
        'Answer': '',
        'Confidence': ''
    })

# 101-125: Irrelevant support
for idx, row in data.iterrows():
    question = row['question']

    expanded_data.append({
        'Question': question,
        'Description': "Provide your best guess and the probability that it is correct (1 to 10) for the following question based on the support.",

        'Support': row['irr_support'],
        'Answer': '',
        'Confidence': ''
    })

# 126-150: Contradictory support
for idx, row in data.iterrows():
    question = row['question']

    expanded_data.append({
        'Question': question,
        'Description': "Provide your best guess and the probability that it is correct (1 to 10) for the following question based on the support.",

        'Support': row['contradict_support'],
        'Answer': '',
        'Confidence': ''
    })

# 151-175: Incomplete support
for idx, row in data.iterrows():
    question = row['question']

    expanded_data.append({
        'Question': question,
        'Description': "Provide your best guess and the probability that it is correct (1 to 10) for the following question based on the support.",

        'Support': row['incomplete_support'],
        'Answer': '',
        'Confidence': ''
    })

# Convert the expanded data into a DataFrame
expanded_df = pd.DataFrame(expanded_data)

# Save the expanded DataFrame as a CSV file
expanded_csv_path = '/userhomes/minsu/EviConf/data/SciQ/Final/Human/human_survey_expanded_full.csv'
expanded_df.to_csv(expanded_csv_path, index=False)