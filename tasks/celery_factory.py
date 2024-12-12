from celery import Celery
from Utils.config import cfg
from urllib.parse import quote

redis_host = cfg.INFO.REDIS_HOST
redis_port = cfg.INFO.REDIS_PORT
redis_pwd = cfg.INFO.REDIS_PWD

PASSWORD = quote(redis_pwd)
CELERY_BROKER_URL = f"redis://:{PASSWORD}@{redis_host}:{redis_port}/7"
CELERY_RESULT_BACKEND = f"redis://:{PASSWORD}@{redis_host}:{redis_port}/8"


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