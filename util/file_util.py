import os
import codecs
from os import path
import csv
from os import listdir
import json
from itertools import product, chain


def get_filename(fie_path):
    return os.path.basename(fie_path)


def get_all_lines(file_path, encoding='utf8'):
    lines = []
    try:
        with codecs.open(file_path, "r", encoding=encoding) as file_stream:
            lines = file_stream.read().splitlines()
    except OSError as err:
        print("OS error: {0}".format(err))
    return lines


def get_string(file_path, encoding='utf8'):
    text = None
    try:
        with codecs.open(file_path, "r", encoding=encoding) as file_stream:
            text = file_stream.read()
    except OSError as err:
        print("OS error: {0}".format(err))
    return text


def write_list_file(ele_list, file_path, newline=True):
    try:
        with codecs.open(file_path, 'w+', encoding="utf8") as file_stream:
            for line in ele_list:
                if newline:
                    file_stream.write(line + '\n')
                else:
                    file_stream.write(line)

            file_stream.close()
    except OSError as err:
        print("OS error: {0}".format(err))


def write_string_to_file(content, file_path):
    try:
        with codecs.open(file_path, 'w+', encoding="utf8") as file_stream:
            file_stream.write(content)
    except OSError as err:
        print("OS error: {0}".format(err))


def get_all_files_sublist(dif_path, extension='all'):
    if extension == 'all':
        return chain.from_iterable([[os.sep.join(w) for w in product([i[0]], i[2])] for i in os.walk(dif_path)])
    else:
        flist = chain.from_iterable([[os.sep.join(w) for w in product([i[0]], i[2])] for i in os.walk(dif_path)])
        return [f for f in flist if get_extension(f) == extension]


def get_file_lists(dir_path):
    file_abs_list = []
    for p in listdir(dir_path):
        file_abs_list.append(path.join(dir_path, p))

    return file_abs_list;


def file_exist(file_path):
    isfile = path.exists(file_path)
    if not isfile:
        print('The file does not existed : '+file_path)
        return False
    else:
        return True


# get file extension
def get_extension(file_path):
    return os.path.splitext(file_path)[1][1:]


# get file name without extension
def get_file_name_without_extension(file_name):
    return os.path.splitext(os.path.basename(file_name))[0]


# get file path without extension
def get_file_path_without_extension(file_path):
    return os.path.splitext(file_path)[0]


# write a list as a csv
def write_list_csv(ele_list, file_path):
    try:
        with open(file_path, 'w', encoding='utf-8', newline='') as csv_stream:
            wr = csv.writer(csv_stream, delimiter='\t')
            for line in ele_list:
                wr.writerow(line)
            csv_stream.close()
    except OSError as err:
        print("OS error: {0}".format(err))


def get_file_path(path):
    if os.path.isfile(path):
        return path
    else:
        None


def count_lines(path, buffer_size=65536):
    path = get_file_path(path)
    if path is None:
         print("File %s not found", path)
         return None
    with open(path, "rb") as f:
        num_lines = 0
        eol = False
        while True:
            data = f.read(buffer_size)
            if not data:
                if not eol:
                    num_lines += 1
                return path, num_lines
            num_lines += data.count(b"\n")
            eol = True if data.endswith(b"\n") else False


def ensure_unicode_string(f):
    with open(f, "rb") as file_stream:
        lines = file_stream.readlines()
        print(len(lines))
        for l in lines:
            if isinstance(l, bytes):
                l = l.decode("utf-8")  # Ensure Unicode string.


def read_json(json_path):
    with open(json_path, encoding='UTF-8') as json_file:
        return json.load(json_file)


if __name__ == '__main__':
    fpath = '../data/test/match_regex_rule.en.json'
    json_data = read_json(fpath)

    for attr, val in json_data.items():
        print(attr, val)