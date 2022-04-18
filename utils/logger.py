import logging
import logging.config
from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent.parent
config= {
    'version' : 1,
    'disable_existing_loggers' : False,
    'formatters': {
        'verbose': {
            'format': '{levelname} {asctime} {module} {process:d} {thread:d} {message}',
            'style': '{',
        },
        'simple': {
            'format': '{levelname} {message}',
            'style': '{',
        },
    },
    'filters' : {
        'require_debug_true' : {
            '()': 'django.utils.log.RequireDebugTrue',
        },
    },
    'handlers': {
        'console' : {
            'level' : 'INFO',
            'class' : 'logging.StreamHandler',
            'formatter' : 'simple',
        },
        'debug': {
            'level': 'DEBUG',
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': os.path.join(BASE_DIR, 'Log/debug.log'),
            'formatter' : 'verbose',
            'maxBytes': 1024 * 1024 * 100,
            'backupCount': 10,  # 备份数
            'encoding': 'utf-8',  # 设置默认编码
        },
        'info': {
            'level': 'INFO',
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': os.path.join(BASE_DIR, 'Log/info.log'),
            'formatter' : 'verbose',
            'maxBytes': 1024 * 1024 * 10,
            'backupCount': 5,  # 备份数
            'encoding': 'utf-8',  # 设置默认编码
        },
        'warn': {
            'level': 'WARNING',
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': os.path.join(BASE_DIR, 'Log/warn.log'),
            'formatter' : 'verbose',
            'maxBytes': 1024 * 1024 * 10,
            'backupCount': 5,  # 备份数
            'encoding': 'utf-8',  # 设置默认编码
        },
    },
    'loggers' : {
        'mh' : {
            'handlers' : ['console', 'debug','info','warn'],
            'level' : 'DEBUG',
        }
    }
}
class Logger:
    def __init__(self,logger_name):
        self.set_dirs_files()
        logging.config.dictConfig(config)
        self.__logger=logging.getLogger(logger_name)

    def set_dirs_files(self):
        import sys
        LOG_PATH = os.path.join(BASE_DIR, 'log')
        if not os.path.exists(LOG_PATH):
            os.makedirs(LOG_PATH)
        DEBUG_FILE = os.path.join(LOG_PATH, 'debug.log')
        WARN_FILE = os.path.join(LOG_PATH, 'warn.log')
        INFO_FILE = os.path.join(LOG_PATH, 'info.log')
        FILES = [DEBUG_FILE, WARN_FILE, INFO_FILE]

        for f in FILES:
            if not os.path.exists(f):
                if str(sys.platform).startswith('win'):
                    with open(f, 'a+') as fp:
                        fp.close()
                else:
                    os.mknod(f)

    def get_log(self):
        return self.__logger

if __name__ == "__main__":
    t=Logger('mh')
    t.get_log().debug("test")