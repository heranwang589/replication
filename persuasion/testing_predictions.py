"""
Testing the accuracy of the GPT model
"""
from data_processing import sample_test, clean_text
from training_gpt import client
from openai import OpenAI
import json
import random
import re
import pandas as pd

tools = [{
    "type": "function",
    "function": {
        "name": "choose_author",
        "description": "pick the author that is successful in persuading the op",
        "parameters": {
            "type": "object",
            "properties": {
                "author_name": {
                    "type": "string",
                    "description": "the name of the author that is successful in persuading the op"
                }
            },
            "required": [
                "author_name"
            ],
            "additionalProperties": False
        },
        "strict": True
    }
}]

def get_author(answer, all_authors):
    sentences = answer.split('.')
    sentences_with_author = {}

    predicted_answer = 'no answer'
    
    for author in all_authors:
        sentences_with_author[author] = []

    for sentence in sentences:
        for author in all_authors:
            if author.lower() in sentence.lower().strip():
                sentences_with_author[author].append(sentence.lower().strip())

    for author, answer in sentences_with_author.items():
        if answer != []:
            predicted_answer = author

    return predicted_answer

def baseline() -> list[dict[str, str]]:
    all_answers = []
    for post in sample_test:
        test = {}
        completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a persuasion analyst for reddit r/changemyview, given a cmv post and two similar responses, you will determine which one is successful in persuading the author of the original post (op), and you will provide and only provide the author of the persuasive post in your answer"}, 
            {"role": "user", "content": f"Op Title: {post['op_title']}\n\nOp: {clean_text(post['op_text'])}\n\nResponse A Author: {post['positive']['author']}\n\nResponse A: {clean_text(post['positive']['comments'][0]['body'])}\n\nResponse B Author: {post['negative']['author']}\n\nResponse B: {clean_text(post['negative']['comments'][0]['body'])}\n\nWhich response is more successful in persuading the op?"}, 
        ],
        tools = tools,
        tool_choice={"type": "function", "function": {"name": "choose_author"}}
        )
        test['op_title'] = post['op_title']
        test['actual_author'] = post['positive']['author']
        function_result = completion.choices[0].message.tool_calls[0]
        gpt_prediction = json.loads(function_result.function.arguments)
        test['prediction_author'] = str(gpt_prediction['author_name'])
        all_answers.append(test)
    return all_answers


def naive_prediction() -> list[dict[str, str]]:
    all_answers = []
    for post in sample_test:
        test = {}
        completion = client.chat.completions.create(
        model="ft:gpt-4o-mini-2024-07-18:personal:cmv-persuasion-predictor:BHNY2Wqw",
        messages=[
            {"role": "system", "content": "You are a persuasion analyst for reddit r/changemyview, given a cmv post and two similar responses, you will determine which one is successful in persuading the author of the original post (op), and you will provide and only provide the author of the persuasive post in your answer"}, 
            {"role": "user", "content": f"Op Title: {post['op_title']}\n\nOp: {clean_text(post['op_text'])}\n\nResponse A Author: {post['positive']['author']}\n\nResponse A: {clean_text(post['positive']['comments'][0]['body'])}\n\nResponse B Author: {post['negative']['author']}\n\nResponse B: {clean_text(post['negative']['comments'][0]['body'])}\n\nWhich response is more successful in persuading the op?"}, 
        ],
        tools = tools,
        tool_choice={"type": "function", "function": {"name": "choose_author"}}
        )
        test['op_title'] = post['op_title']
        test['actual_author'] = post['positive']['author']
        function_result = completion.choices[0].message.tool_calls[0]
        gpt_prediction = json.loads(function_result.function.arguments)
        test['prediction_author'] = str(gpt_prediction['author_name'])
        all_answers.append(test)
    return all_answers


def evaluation(predictions: list[dict[str, str]]) -> dict[str, int]:
    results = {}
    results['correct_prediction_count'] = 0
    results['no_prediction'] = 0
    results['wrong_prediction_count'] = 0
    for case in predictions:
        if case['prediction_author'] == case['actual_author']:
            results['correct_prediction_count'] += 1
        elif case['prediction_author'] == 'no answer':
            results['no_prediction'] += 1
        else:
            results['wrong_prediction_count'] +=1
    assert results['correct_prediction_count'] + results['no_prediction'] + results['wrong_prediction_count'] == len(predictions)
    return results
    

all_pte_explanations = []


def predict_then_explain():
    all_answers = []
    for post in sample_test:
        test = {}
        completion = client.chat.completions.create(
        model="ft:gpt-4o-mini-2024-07-18:personal:cmv-persuasion-predictor:BHNY2Wqw",
        messages=[
            {"role": "system", "content": "You are a persuasion analyst for reddit r/changemyview, given a cmv post and two similar responses, you will determine which one is successful in persuading the author of the original post (op). Your response MUST contain an answer in this exact format at the very start: The author who successfully persuaded the OP is:PREDICTION: [author_name]. After the prediction, you MUST provide a DETAILED explanation of of WHY the author's response is persuasive."}, 
            {"role": "user", "content": f"Op Title: {post['op_title']}\n\nOp: {clean_text(post['op_text'])}\n\nResponse A Author: {post['positive']['author']}\n\nResponse A: {clean_text(post['positive']['comments'][0]['body'])}\n\nResponse B Author: {post['negative']['author']}\n\nResponse B: {clean_text(post['negative']['comments'][0]['body'])}\n\nWhich response is more successful in persuading the op?"}, 
        ],
        )
        test['op_title'] = post['op_title']
        test['actual_author'] = post['positive']['author']
        model_answer = completion.choices[0].message.content
        authors = [post['positive']['author'], post['negative']['author']]
        match = re.search(r"PREDICTION:\s+([A-Za-z0-9_-]+)", model_answer)
        if match:
            test['prediction_author'] = match.group(1)
        else:
            test['prediction_author'] = get_author(model_answer, authors)
        all_answers.append(test)
        all_pte_explanations.append(str(completion.choices[0].message.content))
    return all_answers


all_etp_explanations = []


def explain_then_predict():
    all_answers = []
    for post in sample_test:
        test = {}
        completion = client.chat.completions.create(
        model="ft:gpt-4o-mini-2024-07-18:personal:cmv-persuasion-predictor:BHNY2Wqw",
        messages=[
            {"role": "system", "content": "You are a persuasion analyst for reddit r/changemyview, given a cmv post and two similar responses, you will determine which one is successful in persuading the author of the original post (op). DO NOT include a prediction at the start. You MUST first provide a DETAILED explanation of WHY each author's response is persuasive or not before you make any prediction, THEN state your prediction. Your response MUST contain an answer in this exact format at the very end: The author who successfully persuaded the OP is: PREDICTION: [author_name]"}, 
            {"role": "user", "content": f"Op Title: {post['op_title']}\n\nOp: {clean_text(post['op_text'])}\n\nResponse A Author: {post['positive']['author']}\n\nResponse A: {clean_text(post['positive']['comments'][0]['body'])}\n\nResponse B Author: {post['negative']['author']}\n\nResponse B: {clean_text(post['negative']['comments'][0]['body'])}\n\nWhich response is more successful in persuading the op?"}, 
        ],
        )
        test['op_title'] = post['op_title']
        test['actual_author'] = post['positive']['author']
        model_answer = completion.choices[0].message.content
        authors = [post['positive']['author'], post['negative']['author']]
        match = re.search(r"PREDICTION:\s+([A-Za-z0-9_-]+)", model_answer)
        if match:
            test['prediction_author'] = match.group(1)
        else:
            test['prediction_author'] = get_author(model_answer, authors)
        all_answers.append(test)
        all_etp_explanations.append(str(completion.choices[0].message.content))
    return all_answers

all_analysis_frameworks = []

def create_analyis_framework():
    completion = client.chat.completions.create(
        model="ft:gpt-4o-mini-2024-07-18:personal:cmv-persuasion-predictor:BHNY2Wqw",
        messages=[
            {"role": "system", "content": "You are a persuasion analyst for reddit r/changemyview, you have learned the persuasiveness patterns of posts that is successful in changing the op's view"}, 
            {"role": "user", "content": "from your observations, provide a comprehensive framework of factors that influences persuasivness, use numbered points"}, 
        ],
        )
    all_analysis_frameworks.append(completion.choices[0].message.content)
    return str(completion.choices[0].message.content)

analysis_framework = create_analyis_framework()

all_pwe_explanations = []

def predict_while_explain():
    """
    ask the model to come up with a evaluative system for prediction
    for each test post, make it evaluate using the evaluative system it generated, giving the prediction after each step in the system (and seeing if it changed or not)
    """
    all_answers = []
    for post in sample_test:
        test = {}
        completion = client.chat.completions.create(
        model="ft:gpt-4o-mini-2024-07-18:personal:cmv-persuasion-predictor:BHNY2Wqw",
        messages=[
            {"role": "system", "content": f"You are a persuasion analyst for reddit r/changemyview, given a cmv post and two similar responses, you will determine which one is successful in persuading the author of the original post (op) based on the analysis framework {analysis_framework}, after each numbered point, reevaluate your answer based on all the numbered point before. You MUST provide an explanation as to how each numbered point changed/or did not change your prediction. Your response MUST contain an answer in this exact format at the very end: The author who successfully persuaded the OP is: PREDICTION: [author_name]"}, 
            {"role": "user", "content": f"Op Title: {post['op_title']}\n\nOp: {clean_text(post['op_text'])}\n\nResponse A Author: {post['positive']['author']}\n\nResponse A: {clean_text(post['positive']['comments'][0]['body'])}\n\nResponse B Author: {post['negative']['author']}\n\nResponse B: {clean_text(post['negative']['comments'][0]['body'])}\n\nWhich response is more successful in persuading the op?"}, 
        ],
        )
        test['op_title'] = post['op_title']
        test['actual_author'] = post['positive']['author']
        model_answer = completion.choices[0].message.content
        authors = [post['positive']['author'], post['negative']['author']]
        match = re.search(r"PREDICTION:\s+([A-Za-z0-9_-]+)", model_answer)
        if match:
            test['prediction_author'] = match.group(1)
        else:
            test['prediction_author'] = get_author(model_answer, authors)
        all_answers.append(test)
        all_pwe_explanations.append(completion.choices[0].message.content)
    return all_answers


baseline_count = 0
naive_count = 0
pte_count = 0
etp_count = 0
pwe_count = 0


if __name__ == "__main__":

    # baseline prediction
    baseline_prediction_results = baseline()
    #random_sample = random.randint(0, len(baseline_prediction_results) - 1)
    baseline_prediction_eval = evaluation(baseline_prediction_results)
    baseline_count += baseline_prediction_eval['correct_prediction_count']

    # naive prediction
    naive_prediction_results = naive_prediction()
    naive_prediction_eval = evaluation(naive_prediction_results)
    naive_count += naive_prediction_eval['correct_prediction_count']

    # predict then explain
    predict_then_explain_results = predict_then_explain()
    predict_then_explain_eval = evaluation(predict_then_explain_results)
    pte_count += predict_then_explain_eval['correct_prediction_count']

    # explain then predict
    explain_then_predict_results = explain_then_predict()
    explain_then_predict_eval = evaluation(explain_then_predict_results)
    etp_count += explain_then_predict_eval['correct_prediction_count']

    # predict while explain
    predict_while_explain_results = predict_while_explain()
    predict_while_explain_eval = evaluation(predict_while_explain_results)
    pwe_count += predict_while_explain_eval['correct_prediction_count']

    # print debugs

    """
    baseline_prediction_results = baseline()
    baseline_prediction_eval = evaluation(baseline_prediction_results)
    print(f"baseline prediction produced: {baseline_prediction_eval['correct_prediction_count']} correct predictions, {baseline_prediction_eval['wrong_prediction_count']} wrong predictions, and {baseline_prediction_eval['no_prediction']} format failures")
    print("______________________")

    naive_prediction_results = naive_prediction()
    naive_prediction_eval = evaluation(naive_prediction_results)
    print(f"naive prediction produced: {naive_prediction_eval['correct_prediction_count']} correct predictions, {naive_prediction_eval['wrong_prediction_count']} wrong predictions, and {naive_prediction_eval['no_prediction']} format failures")
    print("______________________")

    predict_then_explain_results = predict_then_explain()
    random_sample = random.randint(0, len(predict_then_explain_results) - 1)
    predict_then_explain_eval = evaluation(predict_then_explain_results)
    print(f"predict then explain produced: {predict_then_explain_eval['correct_prediction_count']} correct predictions, {predict_then_explain_eval['wrong_prediction_count']} wrong predictions, and {predict_then_explain_eval['no_prediction']} format failures")
    print(f"random sample: {predict_then_explain_results[random_sample]}")
    print(f"gpt answer: {all_pte_explanations[random_sample]}")
    print("______________________")

    explain_then_predict_results = explain_then_predict()
    explain_then_predict_eval = evaluation(explain_then_predict_results)
    print(f"explain then predict produced: {explain_then_predict_eval['correct_prediction_count']} correct predictions, {explain_then_predict_eval['wrong_prediction_count']} wrong predictions, and {explain_then_predict_eval['no_prediction']} format failures")
    print(f"random sample: {explain_then_predict_results[random_sample]}")
    print(f"gpt answer: {all_etp_explanations[random_sample]}")
    print("______________________") 

    print(analysis_framework)

    predict_while_explain_results = predict_while_explain()
    random_sample = random.randint(0, len(predict_while_explain_results) - 1)
    predict_while_explain_eval = evaluation(predict_while_explain_results)
    print(f"predict while explain produced: {predict_while_explain_eval['correct_prediction_count']} correct predictions, and {predict_while_explain_eval['wrong_prediction_count']} wrong predictions, and {predict_while_explain_eval['no_prediction']} format failures")
    print(f"random sample: {predict_while_explain_results[random_sample]}")
    print(f"gpt answer: {all_pwe_explanations[random_sample]}")
    print("______________________")

    """

    all_results = pd.DataFrame()

    all_results['op_title'] = [result['op_title'] for result in naive_prediction_results]

    all_results['actual_author'] = [result['actual_author'] for result in naive_prediction_results]

    all_results['baseline_prediction'] = [result['prediction_author'] for result in baseline_prediction_results]
    all_results['naive_prediction'] = [result['prediction_author'] for result in naive_prediction_results]
    all_results['predict_then_explain'] = [result['prediction_author'] for result in predict_then_explain_results]
    all_results['explain_then_predict'] = [result['prediction_author'] for result in explain_then_predict_results]
    all_results['predict_0-while_explain'] = [result['prediction_author'] for result in predict_while_explain_results]

    all_results['predict_then_explain_message'] = all_pte_explanations
    all_results['explain_then_predict_message'] = all_etp_explanations
    all_results['predict_while_explain_message'] = all_pwe_explanations

    all_results.to_csv('persuasion_prediction_results.csv', index=False)

    #df = pd.read_csv('persuasion_prediction_results.csv')
    #print(df.head())