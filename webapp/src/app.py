import flask
from flask_cors import CORS, cross_origin
from generator import Generator

ge = Generator()
app = flask.Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


@app.route('/')
@cross_origin()
def api_exec():
    """
    API execute
    """

    input = flask.request.args.get('input')
    temperature = float(flask.request.args.get('temperature', default=0.8))
    top_p = float(flask.request.args.get('top_p',  default=0.9))
    output = ge.generate(input, temperature=temperature, top_p=top_p)

    return flask.jsonify({"result": output})

if __name__ == "__main__":
    app.run()
