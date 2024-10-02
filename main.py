import logging
from flask import Flask, jsonify
from tasks.dispatcher_worker import bp_tasks
from tasks.dispatcher_worker import bp_tasks
from Module01.module01_flask import module01
from Module02.module02_flask import module02
from Flood.flood_flask import floodCalc


app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False
app.register_blueprint(bp_tasks, url_prefix='/tasks')
app.register_blueprint(module01, url_prefix='/module01')
app.register_blueprint(module02, url_prefix='/module02')
app.register_blueprint(floodCalc, url_prefix='/floodCalc')

# app.register_blueprint(module03, url_prefix='/module03')
# app.register_blueprint(module04, url_prefix='/module04')
# app.register_blueprint(module05, url_prefix='/module05')
# app.register_blueprint(module06, url_prefix='/module06')
# app.register_blueprint(module07, url_prefix='/module07')
# app.register_blueprint(module08, url_prefix='/module08')
# app.register_blueprint(module09, url_prefix='/module09')
# app.register_blueprint(module10, url_prefix='/module10')
# app.register_blueprint(module11, url_prefix='/module11')
# app.register_blueprint(module13, url_prefix='/module13')


# 跨域支持
def after_request(resp):
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp


app.after_request(after_request)


@app.errorhandler(500)
def bad_request(error):
    response = {'code': 500, 'msg': str(error.original_exception), 'data': {}}
    # return jsonify({"msg": "Bad Request", "status": 400}), 400
    return jsonify(response)


@app.before_request
def process_request():
    # request session redirect render_template
    # print("所有请求之前都会执行这个函数")
    pass


if __name__ != '__main__':
    gunicorn_logger = logging.getLogger('gunicorn.error')
    app.logger.handlers = gunicorn_logger.handlers
    app.logger.setLevel(gunicorn_logger.level)
    logging.basicConfig(format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s', datefmt='%a, %d %b %Y %H:%M:%S')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, debug=False)
