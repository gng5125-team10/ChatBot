#load_df.py
import pandas as pd

df = pd.read_pickle("./data/loneliness_forum_data_page2_50.pkl")
df1 = pd.read_pickle("./data/loneliness_forum_data_50_100.pkl")

df = df.append(df1)

print(df)


df_questions_only = pd.DataFrame()
df_questions_only['Posts'] = df['Posts'].values
df_questions_only = df_questions_only.drop_duplicates()
print(len(df_questions_only))

print(df_questions_only.head())

##select random 1000 rows
df_questions_only = df_questions_only.sample(n=1000)
df_questions_only.to_excel("Random1000_questions.xlsx", engine='xlsxwriter')