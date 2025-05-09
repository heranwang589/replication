import json
import tarfile
from bz2 import BZ2File
import random

# process and open data

tar = tarfile.open('../cmv.tar.bz2', 'r')

train = tar.extractfile('op_task/train_op_data.jsonlist.bz2')
test = tar.extractfile('op_task/heldout_op_data.jsonlist.bz2')

deserialized_train = [
    json.loads(line.decode('utf-8'))
    for line in BZ2File(train)
]

deserialized_test = [
    json.loads(line.decode('utf-8'))
    for line in BZ2File(test)
]

# random sample

sample_train = random.sample(deserialized_train, 500)
sample_test = random.sample(deserialized_test, 500)



# helper functions

def clean_text(text):
    """
    helper function to remove html formatting tags from text
    """
    lines = [line for line in text.splitlines()
             if not line.lstrip().startswith("&gt;")
             and not line.lstrip().startswith("____")
             and "edit" not in " ".join(line.lower().split()[:2])
            ]
    return "\n".join(lines)

def show_comment(comment):
    """
    helper function to print out post, for debuggint purposes
    """
    text = f"Title: {comment['title']}\n\nText: {comment['selftext']}\n\nDelta:{comment['delta_label']}"
    return clean_text(text)

# format training data for GPT API integration

training_data = []

for post in sample_train:
    training_data.append({
        "messages": [
            {"role": "system", "content": "You are a persuasion resistance analyst for reddit r/changemyview, given a cmv post, you will determine whether the op (original poster) is someone that could be successfully persuaded or not."}, 
            {"role": "user", "content": f"Title: {post['title']}\n\nComment: {clean_text(post['selftext'])}\n\nAuthor: {post['name']}\n\nIs this author malleable to persuasion?"}, 
            {"role": "assistant", "content": f"The malleability status of this author is: {post['delta_label']}."}
        ]
    })

with open("training_data.jsonl", "w") as f:
    for post in training_data:
        f.write(json.dumps(post) + "\n")

