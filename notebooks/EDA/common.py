"""

common functions

"""

import string
from scipy.stats import ttest_ind
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))


def count_stop_words(df, column_name, _stop_words):
    """
    count the number of stop words in a column.
    :param df: dataframe
    :param column_name: column name
    :param _stop_words: stop words
    :return: results, a series of the count of stop words
    """
    results = df[column_name].apply(lambda x: len([word for word in x.split() if word.lower() in _stop_words]))
    return results

def count_punctuation(df, column_name):
    """
    count the amount of punctuation in a column.
    :param df: dataframe
    :param column_name: column name
    :return: results, a series of the count of punctuation
    """
    results = df[column_name].apply(lambda x: len([word for word in x if word in string.punctuation]))
    return results

def count_uppercase(df, column_name):
    """
    count the number of uppercase letters in a column.
    :param df: dataframe
    :param column_name: column name
    :return: results, a series of the count of uppercase letters
    """
    results = df[column_name].apply(lambda x: len([word for word in x if word.isupper()]))
    return results

def count_lowercase(df, column_name):
    results = df[column_name].apply(lambda x: len([word for word in x if word.islower()]))
    return results

def count_words(df, column_name):
    results = df[column_name].apply(lambda x: len(x.split()))
    return results

def t_test(df, target_column_name, feature_name, group_1_value, group_2_value):
    """
    perform a t-test to determine if there is a significant difference between the groups.
    :param df: dataframe
    :param target_column_name: target column name
    :param feature_name:  features column name
    :param group_1_value: group 1 value
    :param group_2_value: group 2 value
    :return: None
    """
    group1 = df[df[target_column_name] == group_1_value][feature_name]
    group2 = df[df[target_column_name] == group_2_value][feature_name]
    # Perform the t-test
    t_stat, p_value = ttest_ind(group1, group2)

    print(f"T-test statistic: {t_stat}")
    print(f"P-value: {p_value}")

    # Interpretation
    if p_value < 0.05:
        # print(results.format("a"))
        print(f"There is a significant difference between the groups, indicating a correlation between '{target_column_name}' and '{feature_name}'.")
    else:
        # print(results.format("no"))
        print(f"There is no significant difference between the groups, indicating no correlation between '{target_column_name}' and '{feature_name}'.")
