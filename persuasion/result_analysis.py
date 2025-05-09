import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import dataframe_image as dfi
import string

df = pd.read_csv('persuasion_prediction_results.csv')

# see the percentage success of each model

total_samples = len(df)
baseline_count = sum(df['baseline_prediction'] == df['actual_author'])
naive_count = sum(df['naive_prediction'] == df['actual_author'])
pte_count = sum(df['predict_then_explain'] == df['actual_author'])
etp_count = sum(df['explain_then_predict'] == df['actual_author'])
pwe_count = sum(df['predict_while_explain'] == df['actual_author'])

baseline_success_percentage = (baseline_count / total_samples) * 100
naive_success_percentage = (naive_count / total_samples) * 100
pte_success_percentage = (pte_count / total_samples) * 100
etp_success_percentage = (etp_count / total_samples) * 100
pwe_success_percentage = (pwe_count / total_samples) * 100

print('Persuasion Prediction Summary')

print(f"The success rate of a completely untrained chat-gpt-4o-mini model is: {baseline_success_percentage:.2f}%")
print(f"The success rate of a naive prediction model (no explanation, function call) is: {naive_success_percentage:.2f}%")
print(f"The success rate of a predict-then-explain model is: {pte_success_percentage:.2f}%")
print(f"The success rate of a explain-then-predict model is: {etp_success_percentage:.2f}%")
print(f"The success rate of a predict-while-explain model is: {pwe_success_percentage:.2f}%")

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

success_filter_pte = df['predict_then_explain'] == df['actual_author']
success_filter_etp = df['explain_then_predict'] == df['actual_author']

pte_success_answers = df[success_filter_pte]['predict_then_explain_message'].tolist()
etp_success_answers = df[success_filter_etp]['explain_then_predict_message'].tolist()

print('pte success answers most common words')
pte_success_mcw = analyze_most_common_words(pte_success_answers).iloc[10:30].reset_index(drop=True)
print(pte_success_mcw)

print('etp success answers most common words')
etp_success_mcw = analyze_most_common_words(etp_success_answers).iloc[10:30].reset_index(drop=True)
print(etp_success_mcw)

# bow analysis for incorrect predictions

failure_filter_pte = df['predict_then_explain'] != df['actual_author']
failure_filter_etp = df['explain_then_predict'] != df['actual_author']

pte_failure_answers = df[failure_filter_pte]['predict_then_explain_message'].tolist()
etp_failure_answers = df[failure_filter_etp]['explain_then_predict_message'].tolist()

print('pte failure answers most common words')
pte_failure_mcw = analyze_most_common_words(pte_failure_answers).iloc[10:30].reset_index(drop=True)
print(pte_failure_mcw)

print('etp failure answers most common words')
etp_failure_mcw = analyze_most_common_words(etp_failure_answers).iloc[10:30].reset_index(drop=True)
print(etp_failure_mcw)

# analysis to see if there is some that they all got wrong

"""
this is to see if there is some trial where all models got the prediction wrong.
after running this, it shows that there isn't, so I settled for more than half
got it wrong
all_fail = ((df['baseline_prediction'] != df['actual_author']) 
            & (df['predict_then_explain'] != df['actual_author']) 
            & (df['explain_then_predict'] != df['actual_author'])  
            & (df['naive_prediction'] != df['actual_author'])  
            & (df['predict_while_explain'] != df['actual_author']))

incorrect_answers = df[all_fail][['op_title', 'actual_author']]
"""

df['num_wrong_models'] = (
    (df['baseline_prediction'] != df['actual_author']).astype(int) +
    (df['predict_then_explain'] != df['actual_author']).astype(int) +
    (df['explain_then_predict'] != df['actual_author']).astype(int) +
    (df['naive_prediction'] != df['actual_author']).astype(int) +
    (df['predict_while_explain'] != df['actual_author']).astype(int)
)

all_fail = df['num_wrong_models'] >= 3

print('trials where more than half of the models (>=3) failed')
incorrect_answers = df[all_fail][['op_title', 'actual_author', 'num_wrong_models']].reset_index(drop=True)
print(incorrect_answers)

"""
if __name__ == "__main__":
    dfi.export(pte_success_mcw, 'most common words in successful pte.png')
    dfi.export(etp_success_mcw, 'most common words in successful etp.png')
    dfi.export(pte_failure_mcw, 'most common words in failed pte.png')
    dfi.export(etp_failure_mcw, 'most common words in failed etp.png')
    dfi.export(incorrect_answers, 'incorrect answers most.png')
"""