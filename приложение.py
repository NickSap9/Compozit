import flask
from flask import Flask, render_template
import pickle

app = Flask(__name__, template_folder='templates')


@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        return render_template('main.html')

    if flask.request.method == 'POST':
        with open('dnn_model.pkl', 'rb') as f:
            loaded_model = pickle.load(f)
        exp = float(flask.request.form['Потребление смолы, г/м2'])
        y_pred = loaded_model.predict([[exp]])

        return render_template("main.html", result = y_pred)


if __name__ == '__main__':
    app.run()
