from configparser import ConfigParser
from sshtunnel import SSHTunnelForwarder
from Utils.config import cfg


# 读取服务器配置
def get_server_sshtunnel():
    '''
    通过sshtunnel连接服务器数据库 (服务器信息默认配置好)
    '''
    path = cfg.FILES.server_conf
    conf = ConfigParser()
    conf.read(path)

    server = SSHTunnelForwarder(ssh_address_or_host=(conf['server_info']['host'], int(conf['server_info']['port'])),
                                ssh_username=conf['server_info']['username'],
                                ssh_password=conf['server_info']['password'],
                                remote_bind_address=(conf['database_info']['host'], int(conf['database_info']['port'])))  # 设置数据库服务地址及端口

    return server
