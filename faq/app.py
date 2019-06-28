from flask import Flask, request, Response, render_template
from flask_bootstrap import Bootstrap

from faq import *

app = Flask(__name__)
Bootstrap(app)
encoder = BertEncoder()
model = FAQRanking(encoder, "final/ans.npy", "final/ans.pkl", "final/model.json", "final/model.h5")
questions = []
answers = []
count = 0


@app.route('/')
def show_chat():
    global count, questions, answers
    count = 0
    questions = []
    answers = []
    return render_template("chat1landing.html")


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    global count, questions, answers
    if request.method == 'GET':
        question = request.args.get('question')
    elif request.method == 'POST':
        # question = request.json['challenge']
        question = request.form['question']
    with open("q_log.txt", "a+") as f:
        f.write(question + "\n")

    prediction = model.answer(question)
    # prediction = "\n".join(prediction)
    questions.append(question)
    answers.append(prediction)
    count += 1
    if count == 2:
        return render_template("chat2results.html",
                               question1=questions[0], answer1=answers[0],
                               question2=questions[1], answer2=answers[1])
    elif count == 3:
        return render_template("chat3results.html",
                               question1=questions[0], answer1=answers[0],
                               question2=questions[1], answer2=answers[1],
                               question3=questions[2], answer3=answers[2])
    elif count == 4:
        return render_template("chat4results.html",
                               question1=questions[0], answer1=answers[0],
                               question2=questions[1], answer2=answers[1],
                               question3=questions[2], answer3=answers[2],
                               question4=questions[3], answer4=answers[3])
    elif count == 5:
        return render_template("chat5results.html",
                               question1=questions[0], answer1=answers[0],
                               question2=questions[1], answer2=answers[1],
                               question3=questions[2], answer3=answers[2],
                               question4=questions[3], answer4=answers[3],
                               question5=questions[4], answer5=answers[4])
    count %= 5
    questions = [question]
    answers = [prediction]
    return render_template("chat1results.html", question=question, answer=prediction)


@app.route('/health')
def health_check():
    return Response("", status=200)


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
