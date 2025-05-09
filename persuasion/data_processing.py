import json
import tarfile
from bz2 import BZ2File
import random

# process and open data

tar = tarfile.open('../cmv.tar.bz2', 'r')

train = tar.extractfile('pair_task/train_pair_data.jsonlist.bz2')
test = tar.extractfile('pair_task/heldout_pair_data.jsonlist.bz2')

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

def show_post(cmv_post):
    """
    helper function to print out post, for debuggint purposes
    """
    text = f"Title: {cmv_post['op_title']}\n\nText: {cmv_post['op_text']}"
    return clean_text(text)

# format training data for GPT API integration

training_data = []

for post in sample_train:
    training_data.append({
        "messages": [
            {"role": "system", "content": "You are a persuasion analyst for reddit r/changemyview, given a cmv post and two similar responses, you will determine which one is successful in persuading the author of the original post (op)"}, 
            {"role": "user", "content": f"Op Title: {post['op_title']}\n\nOp: {clean_text(post['op_text'])}\n\nResponse A Author: {post['positive']['author']}\n\nResponse A: {clean_text(post['positive']['comments'][0]['body'])}\n\nResponse B Author: {post['negative']['author']}\n\nResponse B: {clean_text(post['negative']['comments'][0]['body'])}Which response is more successful in persuading the op?"}, 
            {"role": "assistant", "content": f"The author {post['positive']['author']}'s response is successful in persudaing the op in my analysis."}
        ]
    })

with open("training_data.jsonl", "w") as f:
    for post in training_data:
        f.write(json.dumps(post) + "\n")

