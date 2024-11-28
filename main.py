import pandas as pd
import math
from sklearn.model_selection import train_test_split
from bag_of_words_a1 import calculate_log_likelihood
from bag_of_words_a1 import classify_review
from data_collection import AI_GENERATED, HUMAN
from utils import convert_text_files_to_df
from utils import vocab_dictionary


if __name__ == '__main__':
    ai_generated_df = convert_text_files_to_df("./data/ai_generated", AI_GENERATED)
    human_df = convert_text_files_to_df("./data/human", HUMAN)
    
    df_overall = pd.concat([ai_generated_df, human_df], ignore_index=True)

    df_train, df_test = train_test_split(df_overall, test_size=0.2, random_state=42)

    full_vocab = vocab_dictionary(df_train)
    # print(len(full_vocab))
    ai_generated_vocab = vocab_dictionary(df_train[df_train[
                                                   'label'] == AI_GENERATED])
    human_vocab = vocab_dictionary(df_train[df_train[
                                                   'label'] == HUMAN])
    # print(len(human_vocab))
    # print(len(ai_generated_vocab))
    # print(df_train.head())

    ai_generated_prior = len(df_train[df_train['label'] == AI_GENERATED]) / len(df_train)  # calculate ai_generated prior
    human_prior = len(df_train[df_train['label'] == HUMAN]) / len(df_train)  # calculate human prior
    # print(human_prior)
    # print(ai_generated_prior)

    likelihood = calculate_log_likelihood(full_vocab, human_vocab, ai_generated_vocab)

    assert round(sum([math.exp(likelihood[word]['human']) for word in
                      likelihood])) == 1, "There is probably a bug calculating the human log likelihood"
    assert round(sum([math.exp(likelihood[word]['ai_generated']) for word in
                      likelihood])) == 1, "There is probably a bug calculating the ai_generated log likelihood"

    num_correct = 0
    for i in range(0, df_test.shape[0]):
        predicted_classification, classification_scores = classify_review(df_test.iloc[i]['text'], likelihood,
                                                                human_prior, ai_generated_prior)
        # print(predicted_classification)
        # print(classification_scores)
        if predicted_classification == df_test.iloc[i]['label']:
            num_correct += 1

    print(num_correct / df_test.shape[0])

    # print(predicted_classification)
    # print(classification_scores)
    # print(df_test.iloc[0]['label'])
