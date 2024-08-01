from Utils.config import cfg

redis_host = cfg.INFO.REDIS_HOST
redis_port = cfg.INFO.REDIS_PORT
redis_pwd = cfg.INFO.REDIS_PWD
redis_info = 'redis://:' + redis_pwd + '@' + redis_host + ':' + redis_port + '/1'

# 参数说明 https://docs.celeryq.dev/en/stable/userguide/configuration.html

broker_url = redis_info
result_backend = redis_info
timezone = 'Asia/Shanghai'
enable_utc = True
task_routes = {'tasks.add': 'low-priority'}
task_annotations = {'tasks.add': {'rate_limit': '5/m'}}

task_acks_late = True
task_reject_on_worker_lost = True
worker_concurrency = 4
worker_prefetch_multiplier = 1
worker_max_tasks_per_child = 10

task_serializer = 'json'
result_serializer = 'json'
result_expires = 0
broker_transport_options = {"max_retries": 5, "interval_start": 0, "interval_step": 1, "interval_max": 10}  # 防止redis连不上导致hangs住
broker_connection_retry_on_startup = True
result_accept_content = ['application/json']

# worker_disable_rate_limits = True  # 关闭限速
# worker_max_memory_per_child = 5000000 # 5gb https://www.cnblogs.com/tracydzf/p/15786984.html

# import
# imports = ('jobs.tasks', 'jobs.email', 'jobs.aaa', 'jobs.periodic')
# include=['some_project.tasks']

# Prioritize your tasks!
# CELERY_QUEUES = (
#     Queue('high', Exchange('high'), routing_key='high'),
#     Queue('normal', Exchange('normal'), routing_key='normal'),
#     Queue('low', Exchange('low'), routing_key='low'),
# )
#
# CELERY_DEFAULT_QUEUE = 'normal'
# CELERY_DEFAULT_EXCHANGE = 'normal'
# CELERY_DEFAULT_ROUTING_KEY = 'normal'
#
# CELERY_ROUTES = {
#     # -- HIGH PRIORITY QUEUE -- #
#     'myapp.tasks.check_payment_status': {'queue': 'high'},
#     # -- LOW PRIORITY QUEUE -- #
#     'myapp.tasks.close_session': {'queue': 'low'},
# }

# CELERY_QUEUES = (
#     Queue('default', Exchange('default'), routing_key='default'),
#     Queue('for_task_A', Exchange('for_task_A'), routing_key='for_task_A'),
#     Queue('for_task_B', Exchange('for_task_B'), routing_key='for_task_B'),
# )
#
# CELERY_ROUTES = {
#     'my_taskA': {'queue': 'for_task_A', 'routing_key': 'for_task_A'},
#     'my_taskB': {'queue': 'for_task_B', 'routing_key': 'for_task_B'},
# }

# celery worker -E -l INFO -n workerA -Q for_task_A
# celery worker -E -l INFO -n workerB -Q for_task_B

# schedules
# from datetime import timedelta
#
# beat_schedule = {
#     'printy': {
#         'task': 'printy',
#         'schedule': timedelta(seconds=5),  # 每 5 秒执行一次
#         'args': (8, 2)
#     }
# }

# Sentry
# By default CELERYD_HIJACK_ROOT_LOGGER = True
# Is important variable that allows Celery to overlap other custom logging handlers
# CELERYD_HIJACK_ROOT_LOGGER = False
#
# LOGGING = {
#     'handlers': {
#         'celery_sentry_handler': {
#             'level': 'ERROR',
#             'class': 'core.logs.handlers.CelerySentryHandler'
#         }
#     },
#
#     'loggers': {
#         'celery': {
#             'handlers': ['celery_sentry_handler'],
#             'level': 'ERROR',
#             'propagate': False,
#         },
#     }
# }

# Keep result only if you really need them: CELERY_IGNORE_RESULT = False
# In all other cases it is better to have place somewhere in db
# CELERY_IGNORE_RESULT = True

# shell_commands
# Launch your workers
# celery worker -E -l INFO -n worker.high -Q high
# celery worker -E -l INFO -n worker.normal -Q normal
# celery worker -E -l INFO -n worker.low -Q low

# This worker will accept tasks if for example all other high queue workers are busy
# celery worker -E -l INFO -n worker.whatever

# Use FLOWER to monitor your Celery app `https://github.com/mher/flower`, `https://flower.readthedocs.io/`
# $ pip install flower

# Launch the server and open http://localhost:5555
# $ flower -A some_project --port=5555

# Or, launch from Celery
# $ celery flower -A proj --address=127.0.0.1 --port=5555

# Broker URL and other configuration options can be passed through the standard Celery options
# $ celery flower -A proj --broker=amqp://guest:guest@localhost:5672//
