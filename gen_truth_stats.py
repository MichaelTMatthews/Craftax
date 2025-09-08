import os
from typing import Counter
import json
import gzip
import pickle

DATA_DIR = os.path.join(os.path.dirname(__file__), 'Traces/Test/raw_data')
all_file_lines = []

for filename in os.listdir(DATA_DIR):
    if filename.endswith('.pkl.gz'):
        file_path = os.path.join(DATA_DIR, filename)
        with gzip.open(file_path, 'rb') as f:
            data = pickle.load(f)
        
            # inventory = entry.get('inventory', [])
            all_file_lines.append(data['all_truths'])


#Generate a distribution over the elements in the lists
element_counts = Counter()
for file_lines in all_file_lines:
    element_counts.update(file_lines)

# Print the most common elements and their counts
for element, count in element_counts.most_common():
    print(f"{element}: {count}")

#Plot the distribution as a graph
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.bar(element_counts.keys(), element_counts.values())
plt.xlabel("Elements")
plt.ylabel("Counts")
plt.title("Distribution of Elements")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#Save the unique set of elements to a JSON file with the distributions 
output_data = [{"element": element, "count": count} for element, count in element_counts.items()]
output_path = os.path.join(DATA_DIR, "element_distribution.json")
with open(output_path, "w", encoding="utf-8") as json_file:
    json.dump(output_data, json_file, indent=2, ensure_ascii=False)