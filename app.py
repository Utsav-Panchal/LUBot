from flask import Flask, render_template, request, jsonify
# from flask_cors import CORS

from chatbot_genisys import response


app = Flask(__name__, template_folder="templates")
app.static_folder = 'static'
# CORS(app)

@app.route('/')
def index_get():  # put application's code here
    return render_template('base.html')


@app.route('/predict', methods=['POST'])
def predict():  # put application's code here
    text = request.get_json().get("message")
    # message = {"answer": "test"}
    res = response(text)
    message = {"answer": res}
    return jsonify(message)


if __name__ == '__main__':
    app.run(debug=True)
