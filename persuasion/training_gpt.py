"""
this file is the commands for training and checking the training status of the
gpt model. The training part are commented out to not train the model when called
"""
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

""" 
# uploading training file

def upload_training_file():
    try:
        with open('training_data.jsonl', 'rb') as f:
            response = client.files.create(
                file = f,
                purpose='fine-tune'
            )
        file_id = response.id
        print(f'File uploaded successfully with ID: {file_id}')
        return file_id
    except Exception as e:
        print(f'Error uploading file: {e}')
        return None

file_id = upload_training_file()

# creating fine-tuning job

job = client.fine_tuning.jobs.create(
            training_file=file_id,
            model='gpt-4o-mini-2024-07-18',
            suffix='cmv-persuasion-predictor',
        )

try:
    job
    print(job.id)
except Exception as e:
    print(f"Error creating fine-tuning job: {e}") 

job_id = 'ftjob-bEqz4YBuQIeFg1yo7FxHndvl'

status = client.fine_tuning.jobs.retrieve(job_id).status
print(f"Job status: {status}")
 """