from flask import Flask
from celery import Celery
from Utils.config import cfg
# from celery_config import broker_url

redis_host = cfg.INFO.REDIS_HOST
redis_port = cfg.INFO.REDIS_PORT
redis_pwd = cfg.INFO.REDIS_PWD
redis_info = 'redis://:' + redis_pwd + '@' + redis_host + ':' + redis_port + '/1'

CELERY_BROKER_URL = redis_info
CELERY_RESULT_BACKEND = redis_info


# initialize celery app
def get_celery_app_instance(app):
    celery = Celery(app.import_name, backend=CELERY_RESULT_BACKEND, broker=CELERY_BROKER_URL
                    # backend=app.config['CELERY_RESULT_BACKEND'],
                    # broker=app.config['CELERY_BROKER_URL']
                    )
    # celery.conf.update(app.config)
    celery.config_from_object('Utils.celery_config')

    class ContextTask(celery.Task):

        def __call__(self, *args, **kwargs):
            with app.app_context():
                return self.run(*args, **kwargs)

    celery.Task = ContextTask
    return celery


def make_celery():
    celery = Celery('app.qhbh', backend=CELERY_RESULT_BACKEND, broker=CELERY_BROKER_URL)
    # celery.conf.update(app.config)
    celery.config_from_object('tasks.celery_config')

    # boiler plate to get our tasks running in the app context
    # TaskBase = celery.Task
    #
    # class ContextTask(TaskBase):
    #     abstract = True
    #
    #     def __call__(self, *args, **kwargs):
    #         with app.app_context():
    #             return TaskBase.__call__(self, *args, **kwargs)
    #
    # celery.Task = ContextTask
    return celery


def make_celery_official(app):
    celery = Celery(app.import_name)
    # celery.conf.update(app.config["CELERY_CONFIG"])
    celery.config_from_object('tasks.celery_config')

    class ContextTask(celery.Task):

        def __call__(self, *args, **kwargs):
            with app.app_context():
                return self.run(*args, **kwargs)

    celery.Task = ContextTask
    return celery


# app = Flask(__name__)
# my_celery = get_celery_app_instance(app)
# $ celery -A your_application.celery worker
# $ celery -A tasks.dispatcher_worker worker --loglevel=info
my_celery = make_celery()
