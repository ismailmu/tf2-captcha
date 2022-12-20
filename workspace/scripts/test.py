file_path = 'C:/Users/eeng/Documents/src/python/tensorflow2/custom-object/captcha/workspace/images/validation'

import pathlib
from os import listdir
from os.path import isfile, join
onlyfiles = [f for f in listdir(file_path) if isfile(join(file_path, f))]
print(onlyfiles)

