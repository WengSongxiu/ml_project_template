# -*- coding: UTF-8 -*-
from flask import request
from flask import Flask
app = Flask(__name__)


@app.route('/index', methods=['POST'])
def post():
    return


if __name__ == '__main__':
    app.run(host='0.0.0.0', port='700')
