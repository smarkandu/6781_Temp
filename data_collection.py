from sklearn.model_selection import train_test_split

from utils import convert_text_files_to_df
from utils import vocab_dictionary
import pandas as pd

AI_GENERATED = 1
HUMAN = 0


def get_overall_data_secondary():
    ai_generated_df = convert_text_files_to_df("./data/chatgpt", AI_GENERATED)
    human_df = convert_text_files_to_df("./data/human", HUMAN)

    df_overall = pd.concat([ai_generated_df, human_df], ignore_index=True)

    return df_overall


def get_data_secondary():
    """
    Obtain the given dataset (1 of 2)

    Data Obtained from
    https://github.com/rexshijaku/chatgpt-generated-text-detection-corpus?tab=readme-ov-file
    (Included in this Project)

    :return: full_vocab, ai_generated_vocab, human_vocab, df_train, df_test
    """
    df_overall = get_overall_data1()

    df_train, df_test = train_test_split(df_overall, test_size=0.2, random_state=42)

    full_vocab = vocab_dictionary(df_train)
    # print(len(full_vocab))
    ai_generated_vocab = vocab_dictionary(df_train[df_train[
                                                       'label'] == AI_GENERATED])
    human_vocab = vocab_dictionary(df_train[df_train[
                                                'label'] == HUMAN])

    return full_vocab, ai_generated_vocab, human_vocab, df_train, df_test


def get_overall_data_primary():
    df_overall = pd.read_csv('./data2/AI_Human.csv', on_bad_lines='skip', encoding='utf-8')
    df_overall.columns = ['text', 'label']

    return df_overall


def get_data_primary():
    """
    Obtain the given dataset (2 of 2)

    Data Obtained from
    https://www.kaggle.com/code/syedali110/ai-generated-vs-human-text-95-accuracy/input?select=AI_Human.csv
    It MUST be downloaded and inserted in the following path (not included due to size)
    :return: full_vocab, ai_generated_vocab, human_vocab, df_train, df_test
    """
    df_overall = get_overall_data2()

    df_train, df_test = train_test_split(df_overall, test_size=0.2, random_state=42)

    full_vocab = vocab_dictionary(df_train)
    # print(len(full_vocab))
    ai_generated_vocab = vocab_dictionary(df_train[df_train[
                                                       'label'] == AI_GENERATED])
    human_vocab = vocab_dictionary(df_train[df_train[
                                                'label'] == HUMAN])

    return full_vocab, ai_generated_vocab, human_vocab, df_train, df_test
