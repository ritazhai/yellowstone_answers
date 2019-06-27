import argparse
import pickle

import numpy as np
import pandas as pd
import tensorflow as tf
from bert_serving.client import BertClient
from keras.models import model_from_json

FALL_BACK = "Doesn't answer your question? Sorry I am still learning! Please try asking in another way."


class BertEncoder:
    # def __init__(self):
        # self.bc = BertClient()

    def encode(self, query):
        # emb = self.bc.encode([query])[0]
        emb = np.random.randn(768)
        return emb


class FAQBot:

    def __init__(self, encoder, kb, embs):
        self.encoder = encoder
        self.questions, self.answers = kb
        self.question_vectors = embs

    def topk_similar(self, query_vec, k):
        score = np.sum(query_vec * self.question_vectors, axis=1) / \
                np.linalg.norm(self.question_vectors, axis=1) / \
            np.linalg.norm(query_vec)
        sorted_idx = np.argsort(score)[::-1]
        answers = []
        answer_scores = []
        i = 0
        while len(answers) < k and i < len(sorted_idx):
            if self.answers[sorted_idx[i]] not in answers:
                answers.append(self.answers[sorted_idx[i]])
                answer_scores.append(score[sorted_idx[i]])
            i += 1
        return answers, answer_scores

    def answer(self, query):
        query_vec = self.encoder.encode(query)
        answers, scores = self.topk_similar(query_vec, 1)
        return answers[0]

    @staticmethod
    def from_excel(encoder, fp="ys.xlsx", q_col="question", a_col="answer", embs="embs.npy"):
        df = pd.read_excel(fp)
        questions, answers = df[q_col], df[a_col]
        embs = np.load(embs)
        return FAQBot(encoder, (questions, answers), embs)


class FAQRanking:

    def __init__(self, encoder, ans_emb, ans, model, model_weights):
        self.encoder = encoder
        self.ans_emb = load_answer_embedding(ans_emb)
        self.ans = load_answer_list(ans)
        self.model = load_keras_model(model, model_weights)
        self.ans_dim = len(self.ans_emb)

    def prepare_pred_data(self, question):
        emb = self.encoder.encode(question)
        q_emb = np.array([emb for _ in range(self.ans_dim)])
        # cos_sim = np.array([cos(emb, self.ans_emb[i]) for i in range(self.ans_dim)])
        cos_sim = np.sum(emb * self.ans_emb, axis=1) / np.linalg.norm(self.ans_emb, axis=1) / np.linalg.norm(emb)
        return np.hstack([q_emb, self.ans_emb, cos_sim.reshape(-1, 1)])

    def pred_prob_ranking(self, question):
        pred_arr = self.prepare_pred_data(question)
        with graph.as_default():
            preds = self.model.predict(pred_arr).squeeze()
        ids = np.argsort(preds)[::-1]
        return ids, preds

    def top_k(self, question, k):
        results = []
        ids, preds = self.pred_prob_ranking(question)
        i = 0
        while len(results) < k and i < len(self.ans):
            original_idx = ids[i]
            ans = self.ans[original_idx].strip()
            if ans not in results:
                results.append(ans)
            i += 1
        # for i in range(k):
        #     original_idx = ids[i]
        #     item = {}
        #     item['rank'] = i + 1
        #     item['confidence'] = round(preds[original_idx], 3)
        #     item['answer'] = self.ans[original_idx]
        #     results.append(item)
        return results

    def answer(self, question, k=3):
        answers = self.top_k(question, k)
        answers.append(FALL_BACK)
        return answers


def cos(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def load_keras_model(model_json, model_weights):
    global model
    global graph

    with open(model_json, 'r') as f:
        model = model_from_json(f.read())
    model.load_weights(model_weights)
    graph = tf.get_default_graph()
    return model


def load_answer_embedding(ans_emb):
    ans = np.load(ans_emb)
    return ans


def load_answer_list(ans_list):
    with open(ans_list, 'rb') as f:
        ans = pickle.load(f)
    return ans


def get_args():
    """parse command line"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--knowledge_data", default="ys.xlsx",
                        help="path to the question answer file")
    parser.add_argument("--question_col", default="question",
                        help="name of the question column")
    parser.add_argument("--answer_col", default="answer",
                        help="name of the answer column")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    encoder = BertEncoder()
    # bot = FAQBot.from_excel(encoder, args.knowledge_data, args.question_col, args.answer_col)
    bot = FAQRanking(encoder, "unique_ans_embs.npy", "ans_list.pkl", "nn_sim_1537.json", "nn_sim_1537.h5")
    while True:
        query = input("\ntype you question:\n")
        if query == "exit":
            print("exit bot...")
            break
        ans = bot.answer(query, 3)
        for i in ans:
            print(i)
