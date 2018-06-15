import logging
import time

logger = None

def initLogging():
    global logger
    if not logger:
        filename = 'logs/run_{:s}.log'.format(time.strftime('%Y%m%d-%H%M'))
        logger = logging.getLogger('logDiz')
        logger.setLevel(logging.INFO)
        ch = logging.FileHandler(filename)
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter('[%(asctime)s] %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)

def LOG(msg, printLog=False):
    global logger
    if not logger:
        initLogging()
    logger.info(msg)
    logger.handlers[0].flush()
    if printLog:
        print(msg)
    

def DEBUG(msg):
    global logger
    if not logger:
        initLogging()
    logger.debug(msg)
    logger.handlers[0].flush()
