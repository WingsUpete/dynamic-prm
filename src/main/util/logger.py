"""
Logger that logs to both stdout and a file
"""
import logging
from datetime import datetime
import sys
import os


class Logger:
    def __init__(self, activate=True, logging_folder='log/', comment=None):
        self.activate = activate
        self.time_tag = datetime.now().strftime("%Y%m%d_%H_%M_%S")
        self.comment = comment
        if self.activate:
            self.logger = logging.getLogger()
            self.logger.setLevel(logging.DEBUG)
            self.logging_folder = logging_folder
            if not os.path.isdir(self.logging_folder):
                os.mkdir(self.logging_folder)

            # Create a File Handler
            self.fName = ('%s_%s.log' % (self.comment, self.time_tag)) if self.comment else ('%s.log' % self.time_tag)

            self.f_handler = logging.FileHandler(os.path.join(self.logging_folder, self.fName))
            self.f_handler.terminator = ''

            # Create an stdout Stream Handler
            self.stdout_handler = logging.StreamHandler(sys.stdout)
            self.stdout_handler.terminator = ''

            self.logger.addHandler(self.f_handler)
            self.logger.addHandler(self.stdout_handler)

    def log(self, message='\nCHECK YOUR CODE\n'):
        if self.activate:
            self.logger.debug(message)

    def close(self):
        if self.activate:
            self.f_handler.close()
            self.stdout_handler.close()
