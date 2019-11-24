from flask import Flask, render_template

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('madlibs.html')


@app.route('/user_fill.html')
def user_fill():
    return render_template('user_fill.html')


@app.route('/computer_fill.html')
def computer_fill():
    return render_template('computer_fill.html')


if __name__ == '__main__':
    app.run(debug=True)
