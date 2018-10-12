# -*- coding: utf-8 -*-
"""
AUTHOR: Yue Peng
EMAIL: yuepeng@sf-express.com
DATE: 2018.08.09

predict question label api
"""
import os
import flask
import sys
from collections import OrderedDict
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
from config import Config
from cores.fengbot.classification.executor import TextClassification

config = Config()
app = flask.Flask(__name__)

tc = TextClassification(task="4")
tc2 = TextClassification(task="42")


@app.route("/hi")
def hi():
    return "Hi, there!"


@app.route("/predict")
def predict():
    data = OrderedDict()
    query = flask.request.args["query"]
    if query == "":
        data["response"] = "No answer"
        data["success"] = False
        return flask.jsonify(data)

    data["response"] = tc.predict(query)

    data["success"] = True

    return flask.jsonify(data)


@app.route("/predict2")
def predict2():
    data = OrderedDict()
    query = flask.request.args["query"]
    if query == "":
        data["response"] = "No answer"
        data["success"] = False
        return flask.jsonify(data)

    data["response"] = tc2.predict(query)

    data["success"] = True

    return flask.jsonify(data)


if __name__ == '__main__':
    app.config["JSON_AS_ASCII"] = False
    # app.run(host="10.118.44.123", port=5002)
    # app.run(host=config.ini["serve"]["hostname"], port=9876)
    app.run(host="127.0.0.1", port=5003)
