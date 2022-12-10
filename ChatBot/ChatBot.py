'''
pip install pandas
pip install sklearn
pip install sentence-transformers

'''

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import random



def aichatbot(text):
    model = SentenceTransformer('jhgan/ko-sroberta-multitask')

    df = pd.read_csv("./ChatBot/ChatbotData.csv")
    df_embeding = pd.read_csv("./ChatBot/embeding.csv", header=None)

    em_result = model.encode(text)
    co_result = []

    for temp in range(len(df_embeding)):
        data = df_embeding.iloc[temp]
        co_result.append(cosine_similarity([em_result],[data])[0][0])

    df['cos'] = co_result
    df_result = df.sort_values('cos', ascending=False)
    r = random.randint(0,5)

    return df_result.iloc[r]['A']

# print(aichatbot('배고파요'))
# print(aichatbot('배고파요'))
# print(aichatbot('배고파요'))
# print(aichatbot('살려주세요'))
# print(aichatbot('살려주세요'))
# print(aichatbot('살려주세요'))
# print(aichatbot('살려주세요'))