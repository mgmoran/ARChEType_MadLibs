from flask import Flask, render_template, request, current_app
from WebHelpers import Database, MadLib, fill_madlib

app = Flask(__name__)


@app.route('/', methods=["GET"])
def home():
    return render_template('madlibs.html')

@app.route('/select_madlib.html', methods=["GET", "POST"])
def select_madlib():
    if request.method == "GET":
        return render_template('select_madlib.html', madlib_titles=app.config["db"].all_madlibs.keys())
    if request.method == "POST":
        title = request.form["madlib"]
        labels = app.config["db"].all_madlibs[title].blanks
        blanks = zip(labels, range(len(labels)))
        return render_template('user_fill.html', title=title, blanks=blanks)

@app.route('/user_fill.html', methods=["GET", "POST"])
def user_fill():
    # if request == "GET":
    #     return render_template('user_fill.html', blanks=blanks)
    if request.method == "POST":
        # FOR THE LOVE OF GOD HANDLE YOUR ERRORS HERE
        response = request.form
        blanks = [y for x, y in response.items() if y != "Submit"]
        reversed_response = {y: x for x, y in response.items()}
        title = reversed_response["Submit"]
        plot = db.all_madlibs[title].plot
        filled = fill_madlib(plot, blanks)
        return render_template('filled_madlib.html', title=title, filled=filled)


@app.route('/computer_fill.html', methods=["GET"])
def computer_fill():
    return render_template('computer_fill.html')


if __name__ == '__main__':
    db = Database("", "")
    title = "Liberty"
    genres = "Comedy Short Family".split(' ')
    plot = "Two escaped convicts (______ & _____) change clothes in the getaway car, but wind up wearing" \
           " each other's pants. The rest of the film involves their trying to exchange pants, in alleys," \
           " in cabs and finally high above the street on the girders of a construction site."
    blanks = ["PROTA", "PROTA"]
    db.add_madlib(MadLib(title, genres, plot, blanks))
    app.config["db"] = db
    app.run(debug=True)
    # app.run(host="0.0.0.0", port=8080)
