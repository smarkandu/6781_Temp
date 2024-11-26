import pandas as pd
import math
from sklearn.model_selection import train_test_split
from bag_of_words_a1 import calculate_log_likelihood
from bag_of_words_a1 import classify_review
from utils import convert_files_to_csv
from utils import vocab_dictionary


if __name__ == '__main__':
    chatgpt_df = convert_files_to_csv("./data/chatgpt", 0)
    human_df = convert_files_to_csv("./data/human", 1)
    
    df_overall = pd.concat([chatgpt_df, human_df], ignore_index=True)

    df_train, df_test = train_test_split(df_overall, test_size=0.2, random_state=42)

    full_vocab = vocab_dictionary(df_train)
    # print(len(full_vocab))
    chatgpt_vocab = vocab_dictionary(df_train[df_train[
                                                   'label'] == 0])
    human_vocab = vocab_dictionary(df_train[df_train[
                                                   'label'] == 1])
    # print(len(human_vocab))
    # print(len(chatgpt_vocab))
    # print(df_train.head())

    chatgpt_prior = len(df_train[df_train['label'] == 0]) / len(df_train)  # calculate chatgpt prior
    human_prior = len(df_train[df_train['label'] == 1]) / len(df_train)  # calculate human prior
    # print(human_prior)
    # print(chatgpt_prior)

    likelihood = calculate_log_likelihood(full_vocab, human_vocab, chatgpt_vocab)

    assert round(sum([math.exp(likelihood[word]['human']) for word in
                      likelihood])) == 1, "There is probably a bug calculating the human log likelihood"
    assert round(sum([math.exp(likelihood[word]['chatgpt']) for word in
                      likelihood])) == 1, "There is probably a bug calculating the chatgpt log likelihood"

    num_correct = 0
    for i in range(0, df_test.shape[0]):
        predicted_sentiment, sentiment_scores = classify_review(df_test.iloc[i]['text'], likelihood,
                                                                human_prior, chatgpt_prior)
        # print(predicted_sentiment)
        # print(sentiment_scores)
        if predicted_sentiment == df_test.iloc[i]['label']:
            num_correct += 1

    print(num_correct / df_test.shape[0])

    # print(predicted_sentiment)
    # print(sentiment_scores)
    # print(df_test.iloc[0]['label'])
