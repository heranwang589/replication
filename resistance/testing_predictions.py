"""
Testing the accuracy of the GPT model
"""
from data_processing import sample_test, clean_text
from training_gpt import client
from openai import OpenAI
import json
import random
import pandas as pd
from pydantic import BaseModel
import tiktoken

class Reason(BaseModel):
    text: str

class PTE_Response(BaseModel):
    prediction: str
    explanation: list[Reason]

class ETP_Response(BaseModel):
    explanation: list[Reason]
    prediction: str

class PWE_Step(BaseModel):
    explanation: str
    prediction: str

class PWE_Response(BaseModel):
    steps: list[PWE_Step]
    final_prediction: str


def calculate_token(message: str) -> int:
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(message))


api_calls_count = 0

tools = [{
    "type": "function",
    "function": {
        "name": "predict_malleability",
        "description": "decide if the op is malleable to persuasion or not",
        "parameters": {
            "type": "object",
            "properties": {
                "malleability_status": {
                    "type": "boolean",
                    "description": "true if the op is malleable, false if not"
                }
            },
            "required": [
                "malleability_status"
            ],
            "additionalProperties": False
        },
        "strict": True
    }
}]


def evaluation(predictions: list[dict[str, str]]) -> dict[str, int]:
    results = {}

    results['correct_prediction_count'] = 0
    results['wrong_prediction_count'] = 0

    for case in predictions:
        if case['prediction_delta'] == case['actual_delta']:
            results['correct_prediction_count'] += 1
        elif case['prediction_delta'] != case['actual_delta']:
            results['wrong_prediction_count'] += 1

    assert results['correct_prediction_count'] + results['wrong_prediction_count'] == len(predictions)
    return results

all_analysis_frameworks = []


def create_analysis_framework():
    completion = client.chat.completions.create(
        model="ft:gpt-4o-mini-2024-07-18:personal:cmv-persuasion-predictor:BHNY2Wqw",
        messages=[
            {"role": "system", "content": "You are a persuasion analyst for reddit r/changemyview, you have learned the patterns of speech in ops (original poster) that is open to persuasion and the patterns of speech in ops that are not"}, 
            {"role": "user", "content": "from your observations, provide a comprehensive framework of the top 5 factors that influences malleability to persuasion, use numbered points"}, 
        ],
        max_tokens=500
        )
    all_analysis_frameworks.append(completion.choices[0].message.content)
    return str(completion.choices[0].message.content)


def baseline():
    all_answers = []
    for post in sample_test:
        test = {}
        completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a persuasion resistance analyst for reddit r/changemyview, given a cmv post, you will determine whether the op (original poster) is someone that could be successfully persuaded or not."}, 
            {"role": "user", "content": f"Title: {post['title']}\n\nComment: {clean_text(post['selftext'])}\n\nAuthor: {post['name']}\n\nIs this author malleable to persuasion?"}, 
        ],
        tools = tools,
        tool_choice={"type": "function", "function": {"name": "predict_malleability"}}
        )
        test['title'] = post['title']
        test['author'] = post['name']
        test['actual_delta'] = str(post['delta_label'])
        function_result = completion.choices[0].message.tool_calls[0]
        gpt_prediction = json.loads(function_result.function.arguments)
        test['prediction_delta'] = str(gpt_prediction['malleability_status'])
        all_answers.append(test)
        global api_calls_count
        api_calls_count += 1
        print(f'{api_calls_count} done / 3000 calls')
    return all_answers


examples = """
Input: Title: Support of government is the support of violence. CMV\n\nComment: Government functions in this way: Me and my colleagues get together in a fancy building and write on a piece of paper that we have the right to steal your money because a bunch of other people voted to outsource their volition to us.\n\nWe call this a 'law', and by calling it 'law' we believe it validates the idea that we have the right to steal from you, but you can not do this to us or anyone else. \n\nIf you do not comply with our demands for your obedience we can also give rights that you don't have to men in costumes wearing badges. These men can go into your home, point a gun at you, and lock you in a cage against your will if you do not comply with what we have declared to be 'law'.\n\nFurthermore, if you have a child on land which we have declared to be under our rule, we will do the same things to coerce them. We will declare the newborns and infants to have entered into a 'social contract' by being born on this land. If they wish to leave this land we will force them to contract with us for the papers which we claim provide the privilege to do this.\n\nNone of this could be enforced without the use of violence. In fact, you can't force anyone to do anything without some form of violence or the threat of violence. CMV\n\nAuthor: t3_1is245\n\nIs this author malleable to persuasion?
Output: The malleability status of this author is: False.

Input: Title: I'm turning 24 and I feel like my youth is over. CMV\n\nComment: So I turn 24 in a couple of months and I already feel like I'm an \"old man\" that has peaked and it's all 50-60 years of downhill from here on out.\n\nI realize that there are other joys I will learn to appreciate later in life, like having kids, being a uncle, even grandparent one day, but I am not ready for those things yet. I don't feel like I have \"lived\" enough of my youth yet as I suffered from a deep depression for the last 2 years.\n\n\nAm I right when I think that at 24, your egoitistical youth days where you can enjoy just being you is over?\n\nAuthor: t3_1abkfe\n\nIs this author malleable to persuasion?
Output: The malleability status of this author is: False.

Input: Title: CMV: Special care should not be taken when dealing with introverts.\n\nComment: Furthermore, I don't believe in a strict dichotomy between introverts and extroverts. I believe people may have MORE introverted tendencies, but to fully identify as an introvert only serves to be self-limiting in social situations. Perhaps this belief contributes to my belief that no additional precautions or care need to be taken when dealing with supposedly introverted people in social situations.  I have seen people who call themselves introverts complaining about being talked to by extroverts and being \"forced to talk\" by being asked questions. I feel that in a social situation it is completely normal to try to talk to everyone in the group and try to hear from everybody. I don't think we should have to put on kid gloves to deal with people who don't like to deal with people. I think this doesn't apply to people who may be autistic because their reason for possibly not liking social situations is because of a disorder and so I have more sympathy for them.\n\n\nAuthor: t3_28a08d\n\nIs this author malleable to persuasion?"
Output: The malleability status of this author is: True.

Input: Title: CMV: We should not have fixed-price fines for unlawful activity\n\nComment: For example, two people are walking the streets of their city and both decide to cross on a red signal.\n\nTwo police officers stop each of them and they are both issued a $50 fine for the infringement. \n\nBoth have steady income and are able to pay the fine however their annual income differs by $80,000.\n\nThe fine for jay-walking is a form of punishment intended to deter an unsafe activity. \n\nThe fine represents an additional subjective risk for the activity when the city requires no monetary recompense.\n\nMy view is that this risk is much lower for the high income earner to the point where it no longer has its intended effect therefore the fixed-price fine is ineffectual.\n\nSince the fine, as a consequence of breaking the law, is no longer a risk, it is an unequal form of punishment and some kind of scale is required.\n\nOne solution may be for fines to be similar to taxes, based on a persons income with a minimum amount.\n\nPeople might also be allowed to choose between a inverse variable amount of community service or the variable fine.\n\nIf these methods were used over fixed-price fines, the risk would be sufficient enough that all people would adhere to the law regardless of their income.\n\nAuthor: t3_2uqd1w\n\nIs this author malleable to persuasion?
Output: The malleability status of this author is: True.
"""


def few_shot_learning():
    all_answers = []
    for post in sample_test:
        test = {}
        completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": f"You are a persuasion resistance analyst for reddit r/changemyview, given a cmv post, you will determine whether the op (original poster) is someone that could be successfully persuaded or not. These are some examples: {examples}"}, 
            {"role": "user", "content": f"Title: {post['title']}\n\nComment: {clean_text(post['selftext'])}\n\nAuthor: {post['name']}\n\nIs this author malleable to persuasion?"}, 
        ],
        tools = tools,
        tool_choice={"type": "function", "function": {"name": "predict_malleability"}}
        )
        test['title'] = post['title']
        test['author'] = post['name']
        test['actual_delta'] = str(post['delta_label'])
        function_result = completion.choices[0].message.tool_calls[0]
        gpt_prediction = json.loads(function_result.function.arguments)
        test['prediction_delta'] = str(gpt_prediction['malleability_status'])
        all_answers.append(test)
        global api_calls_count
        api_calls_count += 1
        print(f'{api_calls_count} done / 3000 calls')
    return all_answers


def naive_prediction():
    all_answers = []
    for post in sample_test:
        test = {}
        completion = client.chat.completions.create(
        model="ft:gpt-4o-mini-2024-07-18:personal:cmv-resistance-predictor:BKady1mB",
        messages=[
            {"role": "system", "content": "You are a persuasion resistance analyst for reddit r/changemyview, given a cmv post, you will determine whether the op (original poster) is someone that could be successfully persuaded or not."}, 
            {"role": "user", "content": f"Title: {post['title']}\n\nComment: {clean_text(post['selftext'])}\n\nAuthor: {post['name']}\n\nIs this author malleable to persuasion?"}, 
        ],
        tools = tools,
        tool_choice={"type": "function", "function": {"name": "predict_malleability"}}
        )
        test['title'] = post['title']
        test['author'] = post['name']
        test['actual_delta'] = str(post['delta_label'])
        function_result = completion.choices[0].message.tool_calls[0]
        gpt_prediction = json.loads(function_result.function.arguments)
        test['prediction_delta'] = str(gpt_prediction['malleability_status'])
        all_answers.append(test)
        global api_calls_count
        api_calls_count += 1
        print(f'{api_calls_count} done / 3000 calls')
    return all_answers


def predict_then_explain():
    all_answers = []
    for post in sample_test:
        test = {}
        test['title'] = post['title']
        test['author'] = post['name']
        test['actual_delta'] = str(post['delta_label'])

        try: 
            completion = client.beta.chat.completions.parse(
            model="ft:gpt-4o-mini-2024-07-18:personal:cmv-resistance-predictor:BKady1mB",
            messages=[
                {"role": "system", "content": "You are a persuasion resistance analyst for reddit r/changemyview... Your response MUST FIRST include a prediction (True/False) and SECOND an explanation with more than two distinct reasons, each explaining different aspects of why the author is or isn't malleable to persuasion."}, 
                {"role": "user", "content": f"Title: {post['title']}\n\nComment: {clean_text(post['selftext'])}\n\nAuthor: {post['name']}\n\nIs this author malleable to persuasion?"}, 
            ],
            response_format=PTE_Response,
            )       
            gpt_prediction = json.loads(completion.choices[0].message.content)
            test['prediction_delta'] = str(gpt_prediction['prediction'])
            test['explanation'] = gpt_prediction['explanation']
        except: 
            final_fallback = client.chat.completions.create(
                    model="ft:gpt-4o-mini-2024-07-18:personal:cmv-resistance-predictor:BKady1mB",
                    messages=[
                        {"role": "system", "content": "You are a persuasion resistance analyst for reddit r/changemyview. Determine if the author is malleable to persuasion, THEN, explain your answer."},  
                        {"role": "user", "content": f"Title: {post['title']}\n\nComment: {clean_text(post['selftext'])}\n\nAuthor: {post['name']}\n\nIs this author malleable to persuasion?"}, 
                    ],
                    tools=tools,
                    tool_choice={"type": "function", "function": {"name": "predict_malleability"}}
                )
            function_result = final_fallback.choices[0].message.tool_calls[0]
            gpt_prediction = json.loads(function_result.function.arguments)
            test['prediction_delta'] = str(gpt_prediction['malleability_status'])
            test['explanation'] = 'final fallback, no explanation'

        all_answers.append(test)
        global api_calls_count
        api_calls_count += 1
        print(f'{api_calls_count} done / 3000 calls')
    return all_answers


def explain_then_predict():
    all_answers = []
    for post in sample_test:
        test = {}
        test['title'] = post['title']
        test['author'] = post['name']
        test['actual_delta'] = str(post['delta_label'])

        try:
            completion = client.beta.chat.completions.parse(
            model="ft:gpt-4o-mini-2024-07-18:personal:cmv-resistance-predictor:BKady1mB",
            messages=[
                {"role": "system", "content": "You are a persuasion resistance analyst for reddit r/changemyview... You MUST FIRST provide more than two distinct reasons explaining different aspects of why the author is or isn't malleable to persuasion, FOLLOWED by your prediction (True/False)."},  
                {"role": "user", "content": f"Title: {post['title']}\n\nComment: {clean_text(post['selftext'])}\n\nAuthor: {post['name']}\n\nIs this author malleable to persuasion?"}, 
            ],
            response_format=ETP_Response,
            )
            gpt_prediction = json.loads(completion.choices[0].message.content)
            test['prediction_delta'] = str(gpt_prediction['prediction'])
            test['explanation'] = gpt_prediction['explanation']
        except: 
            final_fallback = client.chat.completions.create(
                    model="ft:gpt-4o-mini-2024-07-18:personal:cmv-resistance-predictor:BKady1mB",
                    messages=[
                        {"role": "system", "content": "You are a persuasion resistance analyst for reddit r/changemyview. Analyze the post to see if the author is malleable to persuasion, THEN, predict if the author is malleable to persuasion."},  
                        {"role": "user", "content": f"Title: {post['title']}\n\nComment: {clean_text(post['selftext'])}\n\nAuthor: {post['name']}\n\nIs this author malleable to persuasion?"}, 
                    ],
                    tools=tools,
                    tool_choice={"type": "function", "function": {"name": "predict_malleability"}}
                )
            function_result = final_fallback.choices[0].message.tool_calls[0]
            gpt_prediction = json.loads(function_result.function.arguments)
            test['prediction_delta'] = str(gpt_prediction['malleability_status'])
            test['explanation'] = 'final fallback, no explanation'

        all_answers.append(test)
        global api_calls_count
        api_calls_count += 1
        print(f'{api_calls_count} done / 3000 calls')
    return all_answers


analysis_framework = create_analysis_framework()


def predict_while_explain():
    all_answers = []
    for post in sample_test:
        test = {}
        test['title'] = post['title']
        test['author'] = post['name']
        test['actual_delta'] = str(post['delta_label'])

        try: 
            completion = client.beta.chat.completions.parse(
            model="ft:gpt-4o-mini-2024-07-18:personal:cmv-resistance-predictor:BKady1mB",
            messages=[
                {"role": "system", "content": f"You are a persuasion resistance analyst for reddit r/changemyview. Given a post, analyze whether the original poster (OP) is malleable to persuasion based on the following analysis framework: {analysis_framework}\n\nFor your response:\n\n1. Analyze each point in the framework sequentially\n\n2. For EACH numbered point in the framework:\n\n- Provide a detailed explanation of how this point applies to the post\n\n- Include references to specific language, tone, and reasoning from the text\n\n- State your updated prediction after considering this point\n\n- Explain how this point influenced or changed your prediction from previous points\n\nEach step must contain both an explanation and an interim prediction.\n\nAfter analyzing all points, provide your final prediction in this exact format:\n\nThe malleability status of this author is: [True/False]."}, 
                {"role": "user", "content": f"Title: {post['title']}\n\nComment: {clean_text(post['selftext'])}\n\nAuthor: {post['name']}\n\nIs this author malleable to persuasion?"}, 
            ],
            response_format=PWE_Response,
            )
            gpt_prediction = json.loads(completion.choices[0].message.content)
            final_prediction = gpt_prediction['final_prediction']
            if "True" in final_prediction:
                test['prediction_delta'] = 'True'
            else:
                test['prediction_delta'] = 'False'
            test['explanation'] = gpt_prediction['steps']
        except: 
            final_fallback = client.chat.completions.create(
                    model="ft:gpt-4o-mini-2024-07-18:personal:cmv-resistance-predictor:BKady1mB",
                    messages=[
                        {"role": "system", "content": "You are a persuasion resistance analyst for reddit r/changemyview. Predict if the author is malleable to persuasion."},  
                        {"role": "user", "content": f"Title: {post['title']}\n\nComment: {clean_text(post['selftext'])}\n\nAuthor: {post['name']}\n\nIs this author malleable to persuasion?"}, 
                    ],
                    tools=tools,
                    tool_choice={"type": "function", "function": {"name": "predict_malleability"}}
                )
            function_result = final_fallback.choices[0].message.tool_calls[0]
            gpt_prediction = json.loads(function_result.function.arguments)
            test['prediction_delta'] = str(gpt_prediction['malleability_status'])
            test['explanation'] = 'final fallback, no explanation'

        all_answers.append(test)
        global api_calls_count
        api_calls_count += 1
        print(f'{api_calls_count} done / 3000 calls')
    return all_answers

def clean_explanation(explanations: list[dict[str, str]]):
    if isinstance(explanations, list):
        full_ex = [explanation['text'] for explanation in explanations]
        return " ".join(full_ex)
    elif isinstance(explanations, str):
        return explanations
    return "None"

def clean_pwe(explanation):
    if isinstance(explanation, list):
        full_ex = []
        for step in explanation:
            full_ex.append(step['explanation'])
            full_ex.append(step['prediction'])
        return " ".join(full_ex)
    elif isinstance(explanation, str):
        return explanation
    return "None"

def success_rate(count: int) -> int:
    return (count / 500) * 100

baseline_count = 0
few_shot_count = 0
naive_count = 0
pte_count = 0
etp_count = 0
pwe_count = 0

if __name__ == "__main__":
    
    # random_sample = random.randint(0, len(baseline_prediction_results) - 1)

    # predict then explain
    predict_then_explain_results = predict_then_explain()
    predict_then_explain_eval = evaluation(predict_then_explain_results)
    pte_count += predict_then_explain_eval['correct_prediction_count']
    #print(predict_then_explain_results)
    #print(predict_then_explain_eval)
    print(f'predict then explain done, the success rate is {success_rate(pte_count)}%')

    # explain then predict
    explain_then_predict_results = explain_then_predict()
    explain_then_predict_eval = evaluation(explain_then_predict_results)
    etp_count += explain_then_predict_eval['correct_prediction_count']
    #print(explain_then_predict_results)
    #print(explain_then_predict_eval)
    print(f'explain then predict done, the success rate is {success_rate(etp_count)}%')

    # predict while explain
    predict_while_explain_results = predict_while_explain()
    predict_while_explain_eval = evaluation(predict_while_explain_results)
    pwe_count += predict_while_explain_eval['correct_prediction_count']
    #print(predict_while_explain_results)
    #print(predict_while_explain_eval)
    print(f'predict while explain done, the success rate is {success_rate(pwe_count)}%')

    # baseline prediction
    baseline_prediction_results = baseline()
    baseline_prediction_eval = evaluation(baseline_prediction_results)
    #print(baseline_prediction_results)
    #print(baseline_prediction_eval)
    baseline_count += baseline_prediction_eval['correct_prediction_count']
    print(f'baseline predictions done, the success rate is {success_rate(baseline_count)}%')

    # few shot prediction
    few_shot_prediction_results = few_shot_learning()
    few_shot_eval = evaluation(few_shot_prediction_results)
    few_shot_count += few_shot_eval['correct_prediction_count']
    #print(few_shot_prediction_results)
    #print(few_shot_eval)
    print(f'few-shot learning predictions done, the success rate is {success_rate(few_shot_count)}%')

    # naive prediction
    naive_prediction_results = naive_prediction()
    naive_prediction_eval = evaluation(naive_prediction_results)
    naive_count += naive_prediction_eval['correct_prediction_count']
    #print(naive_prediction_results)
    #print(naive_prediction_eval)
    print(f'naive predictions done, the success rate is {success_rate(naive_count)}%')

    # print debugs
    
    #print('_________')
    #print('baseline')
    #print('_________')
    #print(baseline())

    #print('_________')
    #print('fewshot')
    #print('_________')
    #print(few_shot_learning())

    #print('_________')
    #print('naive_prediction')
    #print('_________')
    #print(naive_prediction())

    #print('_________')
    #print('predict_then_explain')
    #print('_________')
    #print(predict_then_explain())

    #print('_________')
    #print('explain then predict')
    #print('_________')
    #print(explain_then_predict())

    #print('_________')
    #print('predict while explain')
    #print('_________')
    #print(predict_while_explain())

    # create csv

    all_results = pd.DataFrame()

    all_results['op_title'] = [result['title'] for result in naive_prediction_results]

    all_results['actual_delta'] = [result['actual_delta'] for result in naive_prediction_results]

    all_results['baseline_prediction'] = [result['prediction_delta'] for result in baseline_prediction_results]
    all_results['few_shot_learning_prediction'] = [result['prediction_delta'] for result in few_shot_prediction_results]
    all_results['naive_prediction'] = [result['prediction_delta'] for result in naive_prediction_results]
    all_results['predict_then_explain'] = [result['prediction_delta'] for result in predict_then_explain_results]
    all_results['explain_then_predict'] = [result['prediction_delta'] for result in explain_then_predict_results]
    all_results['predict_while_explain'] = [result['prediction_delta'] for result in predict_while_explain_results]

    all_results['predict_then_explain_explanation'] = [clean_explanation(result['explanation']) for result in predict_then_explain_results]
    all_results['explain_then_predict_explanation'] = [clean_explanation(result['explanation']) for result in explain_then_predict_results]
    all_results['predict_while_explain_explanation'] = [clean_pwe(result['explanation']) for result in predict_while_explain_results]

    all_results.to_csv('persuasion_prediction_results.csv', index=False)

    with open('frameworks.txt', 'w', encoding='utf-8') as f:
        f.write('\n\n---\n\n'.join(all_analysis_frameworks))