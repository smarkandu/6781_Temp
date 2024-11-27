from sklearn.model_selection import train_test_split

from utils import convert_text_files_to_df
from utils import vocab_dictionary
import pandas as pd

CHATGPT = 1
HUMAN = 0

def get_data1():
    chatgpt_df = convert_text_files_to_df("./data/chatgpt", CHATGPT)
    human_df = convert_text_files_to_df("./data/human", HUMAN)

    df_overall = pd.concat([chatgpt_df, human_df], ignore_index=True)

    df_train, df_test = train_test_split(df_overall, test_size=0.2, random_state=42)

    full_vocab = vocab_dictionary(df_train)
    # print(len(full_vocab))
    chatgpt_vocab = vocab_dictionary(df_train[df_train[
                                                  'label'] == CHATGPT])
    human_vocab = vocab_dictionary(df_train[df_train[
                                                'label'] == HUMAN])

    return full_vocab, chatgpt_vocab, human_vocab, df_train, df_test


def get_data2():
    df_overall = pd.read_csv('./data2/AI_Human.csv')
    df_overall.columns = ['text', 'label']

    df_train, df_test = train_test_split(df_overall, test_size=0.2, random_state=42)

    full_vocab = vocab_dictionary(df_train)
    # print(len(full_vocab))
    chatgpt_vocab = vocab_dictionary(df_train[df_train[
                                                  'label'] == CHATGPT])
    human_vocab = vocab_dictionary(df_train[df_train[
                                                'label'] == HUMAN])

    return full_vocab, chatgpt_vocab, human_vocab, df_train, df_test
