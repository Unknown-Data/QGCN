import logging
import os
import sys
import time

DEFAULT_LOG_FORMAT = ["%(asctime)s", "%(name)s", "%(levelname)s", "%(message)s"]


class BaseLogger(logging.getLoggerClass()):
    # can configure level=logging.[CRITICAL(50), FATAL(CRITICAL), ERROR(40), WARNING(30), WARN(WARNING)
    # INFO(20), DEBUG(10), NOTSET(0 - default)]
    # for various log_format options see help(logging.Formatter)
    def __init__(self, name=None, level=logging.NOTSET, log_format=None):
        if log_format is None:
            log_format = DEFAULT_LOG_FORMAT
            if name is None:
                log_format = log_format[:1] + log_format[2:]
            log_format = " - ".join(log_format)

        if name is None:
            name = type(self).__name__
        super(BaseLogger, self).__init__(name, level=level)

        # create formatter
        self.formatter = logging.Formatter(log_format)

    def _set_format(self, *args, **kwargs):
        self.formatter = logging.Formatter(*args, **kwargs)
        list(map(lambda handler: handler.setFormatter(self.formatter), self.handlers))

    def _initialize_handler(self):
        list(map(lambda handler: handler.setLevel(self.level), self.handlers))

        # attach formatter to handlers
        list(map(lambda handler: handler.setFormatter(self.formatter), self.handlers))

    def close(self):
        logging.shutdown([x.__weakref__ for x in self.handlers])


class FileLogger(BaseLogger):
    def __init__(self, filename, *args, ext="log", path="logs", add_timestamp=False, should_overwrite=True, **kwargs):
        super(FileLogger, self).__init__(*args, **kwargs)

        if not os.path.exists(path):
            os.makedirs(path)

        filename = os.path.join(path, filename)
        if add_timestamp:
            filename += time.strftime("-%Y-%m-%d-%H%M%S")

        mode = 'w' if should_overwrite else 'a'

        self.addHandler(logging.FileHandler("%s.%s" % (filename, ext,), mode=mode))
        self._initialize_handler()


class CSVLogger(FileLogger):
    def __init__(self, *args, **kwargs):
        if "ext" not in kwargs:
            kwargs["ext"] = "csv"
        kwargs["log_format"] = "%(message)s"
        self._delimiter = kwargs.pop("delimiter", ",")
        self._other_del = kwargs.pop("other_delimiter", "-")
        super(CSVLogger, self).__init__(*args, **kwargs)

    def space(self, num_spaces=1):
        self.info("\n" * num_spaces)

    def info(self, *args):
        args = [arg.replace(self._delimiter, self._other_del).replace(" ", "") if self._delimiter in arg else arg
                for arg in map(str, args)]
        super(CSVLogger, self).info(self._delimiter.join(args))


class PrintLogger(BaseLogger):
    def __init__(self, *args, **kwargs):
        super(PrintLogger, self).__init__(*args, **kwargs)

        self.addHandler(logging.StreamHandler(stream=sys.stdout))  # create console handler
        self._initialize_handler()


class EmptyLogger(BaseLogger):
    def __init__(self, *args, **kwargs):
        super(EmptyLogger, self).__init__(*args, **kwargs)

        self.addHandler(logging.NullHandler())
        self._initialize_handler()
        # self.disabled = True


def multi_logger(loggers, name="MultiLogger"):
    res = BaseLogger(name)
    for logger in loggers:
        for handler in logger.handlers:
            res.addHandler(handler)
    return res


if __name__ == "__main__":
    # loggers_list = [FileLogger("temp_file"), PrintLogger("test_me")]
    # i = multi_logger(loggers_list)
    log = PrintLogger()
    log.info("bla")
    print("Bla")

# MetaClass perfect explanation:
# http://stackoverflow.com/questions/100003/what-is-a-metaclass-in-python/6581949#6581949
# class LoggedTestCase(unittest.TestCase):
#    __metaclass__ = LogThisTestCase
#    logger = logging.getLogger("unittestLogger")
#    logger.setLevel(logging.DEBUG) # or whatever you prefer
