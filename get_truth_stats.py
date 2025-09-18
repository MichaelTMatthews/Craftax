import os
from typing import Counter
import json
import gzip
import pickle
import matplotlib.pyplot as plt
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), 'Traces/stone_pickaxe_easy')

# os.makedirs(DATA_DIR + '/groundTruth', exist_ok=True)
# os.makedirs(DATA_DIR + '/actions', exist_ok=True)
os.makedirs(DATA_DIR + '/pixel_obs', exist_ok=True)

all_file_lines = []

for filename in os.listdir(DATA_DIR + '/raw_data'):
    if filename.endswith('.pkl.gz'):
        file_path = os.path.join(DATA_DIR + '/raw_data', filename)
        with gzip.open(file_path, 'rb') as f:
            data = pickle.load(f)
            truths = data['all_truths']
            actions = data['all_actions']
            obs = data['all_obs']

            actions = np.array(actions)
            obs = np.array(obs)

            # np.save(os.path.join(DATA_DIR, 'actions', filename.replace('.pkl.gz', '')), actions)
            np.save(os.path.join(DATA_DIR, 'pixel_obs', filename.replace('.pkl.gz', '')), obs)

#             truth_file_path = os.path.join(DATA_DIR, 'groundTruth', filename.replace('.pkl.gz', ''))
#             with open(truth_file_path, 'w') as truth_file:
#                 for i, truth in enumerate(truths):
#                     if i < len(truths) - 1:
#                         truth_file.write(f"{truth}\n")
#                     else:
#                         truth_file.write(f"{truth}")

#             all_file_lines.append(truths)


# #Generate a distribution over the elements in the lists
# element_counts = Counter()
# for file_lines in all_file_lines:
#     element_counts.update(file_lines)

# unique_skills = set()
# for element, count in element_counts.most_common():
#     print(f"{element}: {count}")
#     unique_skills.add(element)

# plt.figure(figsize=(10, 6))
# plt.bar(element_counts.keys(), element_counts.values())
# plt.xlabel("Elements")
# plt.ylabel("Counts")
# plt.title("Distribution of Elements")
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.savefig(os.path.join(DATA_DIR, "element_distribution.png"))

# #Save the unique set of elements to a JSON file with the distributions 
# output_data = [{"element": element, "count": count} for element, count in element_counts.items()]
# output_path = os.path.join(DATA_DIR, "element_distribution.json")
# with open(output_path, "w", encoding="utf-8") as json_file:
#     json.dump(output_data, json_file, indent=2, ensure_ascii=False)


# os.makedirs(DATA_DIR + '/mapping', exist_ok=True)
# mapping_path = os.path.join(DATA_DIR, 'mapping', "mapping.txt")
# with open(mapping_path, "w", encoding="utf-8") as mapping_file:
#     sorted_skills = sorted(unique_skills)
#     for idx, skill in enumerate(sorted_skills):
#         if idx < len(sorted_skills) - 1:
#             mapping_file.write(f"{idx} {skill}\n")
#         else:
#             mapping_file.write(f"{idx} {skill}")