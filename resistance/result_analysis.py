import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import dataframe_image as dfi
import string

df = pd.read_csv('resistance_prediction_results.csv')

# see the percentage success of each model

total_samples = len(df)
baseline_count = sum(df['baseline_prediction'] == df['actual_delta'])
fewshot_count = sum(df['few_shot_learning_prediction'] == df['actual_delta'])
naive_count = sum(df['naive_prediction'] == df['actual_delta'])
pte_count = sum(df['predict_then_explain'] == df['actual_delta'])
etp_count = sum(df['explain_then_predict'].astype(bool) == df['actual_delta'])
pwe_count = sum(df['predict_while_explain'] == df['actual_delta'])

baseline_success_percentage = (baseline_count / total_samples) * 100
few_shot_success_percentage = (fewshot_count / total_samples) * 100
naive_success_percentage = (naive_count / total_samples) * 100
pte_success_percentage = (pte_count / total_samples) * 100
etp_success_percentage = (etp_count / total_samples) * 100
pwe_success_percentage = (pwe_count / total_samples) * 100

print('Resistance Prediction Summary')

print(f"The success rate of a completely untrained chat-gpt-4o-mini model is: {baseline_success_percentage:.2f}%")
print(f"The success rate of a chat-gpt-4o-mini model with few-shot learning is: {few_shot_success_percentage:.2f}%")
print(f"The success rate of a naive prediction model (no explanation, function call) is: {naive_success_percentage:.2f}%")
print(f"The success rate of a predict-then-explain model is: {pte_success_percentage:.2f}%")
print(f"The success rate of a explain-then-predict model is: {etp_success_percentage:.2f}%")
print(f"The success rate of a predict-while-explain model is: {pwe_success_percentage:.2f}%")

print('Data Cleaning...')

pte_fallback_count = sum(df['predict_then_explain_explanation'] == 'final fallback, no explanation')
pte_no_fallback = df['predict_then_explain_explanation'] != 'final fallback, no explanation'
assert pte_fallback_count + len(df[pte_no_fallback]) == 500
pte_no_fallback_success_count = sum(df[pte_no_fallback]['predict_while_explain'] == df[pte_no_fallback]['actual_delta'])
pte_no_fallback_success_percentage = (pte_no_fallback_success_count / len(df[pte_no_fallback])) * 100
print(f"Predict-then-explain went for fallback {pte_fallback_count} times. The success rate of a predict-then-explain model without fallback is: {pte_no_fallback_success_percentage:.2f}%")

etp_fallback_count = sum(df['explain_then_predict_explanation'] == 'final fallback, no explanation')
etp_no_fallback = df['explain_then_predict_explanation'] != 'final fallback, no explanation'
assert etp_fallback_count + len(df[etp_no_fallback]) == 500
etp_no_fallback_success_count = sum(df[etp_no_fallback]['predict_while_explain'] == df[etp_no_fallback]['actual_delta'])
etp_no_fallback_success_percentage = (etp_no_fallback_success_count / len(df[etp_no_fallback])) * 100
print(f"Explain-then-predict went for fallback {etp_fallback_count} times. The success rate of a explain-then-predict model without fallback is: {etp_no_fallback_success_percentage:.2f}%")

pwe_fallback_count = sum(df['predict_while_explain_explanation'] == 'final fallback, no explanation')
pwe_no_fallback = df['predict_while_explain_explanation'] != 'final fallback, no explanation'
assert pwe_fallback_count + len(df[pwe_no_fallback]) == 500
pwe_no_fallback_count = sum(df[pwe_no_fallback]['predict_while_explain'] == df[pwe_no_fallback]['actual_delta'])
pwe_no_fallback_success_percentage = (pwe_no_fallback_count / len(df[pwe_no_fallback])) * 100
print(f"Predict-while-explain went for fallback {pwe_fallback_count} times. The success rate of a predict-while-explain model without fallback is: {pwe_no_fallback_success_percentage:.2f}%")

def analyze_most_common_words(answers):
    """
    bow analysis for most common words
    """
    ans_cleaned = [answer.lower().translate(str.maketrans('', '', string.punctuation))
               for answer in answers]
    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform(ans_cleaned)
    all_words = vectorizer.get_feature_names_out()
    word_counts = X.toarray().sum(axis=0)
    word_freq_df = pd.DataFrame({'word': all_words, 'count': word_counts})
    count_df = word_freq_df.sort_values(by='count', ascending=False).reset_index(drop=True)
    return count_df



# bow analysis for correct predictions

success_filter_pte = df['predict_then_explain'] == df['actual_delta']
success_filter_etp = df['explain_then_predict'].astype(bool) == df['actual_delta']
success_filter_pwe = df['predict_while_explain'] == df['actual_delta']

pte_success_answers = df[success_filter_pte]['predict_then_explain_explanation'].tolist()
etp_success_answers = df[success_filter_etp]['explain_then_predict_explanation'].tolist()
pwe_success_answers = df[success_filter_pwe]['predict_while_explain_explanation'].tolist()

print('pte success answers most common words')
pte_success_mcw = analyze_most_common_words(pte_success_answers).iloc[3:30].reset_index(drop=True)
print(pte_success_mcw)

print('etp success answers most common words')
etp_success_mcw = analyze_most_common_words(etp_success_answers).iloc[3:30].reset_index(drop=True)
print(etp_success_mcw)

print('pwe success answers most common words')
pwe_success_mcw = analyze_most_common_words(pwe_success_answers).iloc[4:30].reset_index(drop=True)
print(pwe_success_mcw)

# bow analysis for incorrect predictions

failure_filter_pte = df['predict_then_explain'] != df['actual_delta']
failure_filter_etp = df['explain_then_predict'].astype(bool) != df['actual_delta']
failure_filter_pwe = df['predict_while_explain'] != df['actual_delta']

pte_failure_answers = df[failure_filter_pte]['predict_then_explain_explanation'].tolist()
etp_failure_answers = df[failure_filter_etp]['explain_then_predict_explanation'].tolist()
pwe_failure_answers = df[failure_filter_pwe]['predict_while_explain_explanation'].tolist()

print('pte failure answers most common words')
pte_failure_mcw = analyze_most_common_words(pte_failure_answers).iloc[6:30].reset_index(drop=True)
print(pte_failure_mcw)

print('etp failure answers most common words')
etp_failure_mcw = analyze_most_common_words(etp_failure_answers).iloc[5:30].reset_index(drop=True)
print(etp_failure_mcw)

print('pwe failure answers most common words')
pwe_failure_mcw = analyze_most_common_words(pwe_failure_answers).iloc[4:30].reset_index(drop=True)
print(pwe_failure_mcw)

# analysis to see if there is some that they all got wrong


all_fail = ((df['baseline_prediction'] != df['actual_delta']) 
            & (df['predict_then_explain'] != df['actual_delta']) 
            & (df['explain_then_predict'].astype(bool) != df['actual_delta'])  
            & (df['naive_prediction'] != df['actual_delta'])  
            & (df['predict_while_explain'] != df['actual_delta'])
            & (df['few_shot_learning_prediction'] != df['actual_delta']))

incorrect_answers_all = df[all_fail][['op_title', 'actual_delta']]

print('trials where all models failed')
incorrect_answers_most = df[all_fail][['op_title', 'actual_delta']].reset_index(drop=True)
print(incorrect_answers_most)

df['num_wrong_models'] = (
    (df['baseline_prediction'] != df['actual_delta']).astype(int) +
    (df['predict_then_explain'] != df['actual_delta']).astype(int) +
    (df['explain_then_predict'].astype(bool) != df['actual_delta']).astype(int) +
    (df['naive_prediction'] != df['actual_delta']).astype(int) +
    (df['predict_while_explain'] != df['actual_delta']).astype(int) +
    (df['few_shot_learning_prediction'] != df['actual_delta']).astype(int)
)

most_fail = df['num_wrong_models'] > 3

print('trials where more than half of the models (>=3) failed')
incorrect_answers_most = df[most_fail][['op_title', 'actual_delta', 'num_wrong_models']].sort_values(by='num_wrong_models', ascending=False).reset_index(drop=True)
print(incorrect_answers_most)

if __name__ == "__main__":
    dfi.export(pte_success_mcw, 'most common words in successful pte.png')
    dfi.export(etp_success_mcw, 'most common words in successful etp.png')
    dfi.export(pwe_success_mcw, 'most common words in successful pwe.png')

    dfi.export(pte_failure_mcw, 'most common words in failed pte.png')
    dfi.export(etp_failure_mcw, 'most common words in failed etp.png')
    dfi.export(pwe_failure_mcw, 'most common words in failed pwe.png')

    dfi.export(incorrect_answers_all, 'incorrect answers all.png')
    dfi.export(incorrect_answers_most, 'incorrect answers most.png', max_rows=50)
