import os
import json
import gzip
import yaml

def read_gz_file(filepath):
    """
    Read data from gz file
    :param filepath: path of the read file
    :return: json formatted file content.
    """
    with gzip.open(filepath) as reader:
        data_reader = reader.read()
        data_encoded = data_reader.decode('utf-8')
        file_data = json.loads(data_encoded)
        return file_data


def read_txt_file(filepath):
    """
    Read text file into a list of lines
    :param filepath: path of the read file
    :return: list of lines.
    """
    with open(filepath) as reader:
        return [line.replace('\n', '') for line in reader.readlines()]


def read_json_file(filepath):
    """
    Read data from json file
    :param filepath: file path of the json file
    :return: data
    """
    with open(filepath) as reader:
        return json.load(reader)


def load_config():
    config_file = os.path.join('crawler', 'config', 'config.yml')
    return yaml.load(open(config_file), yaml.FullLoader)


if __name__ == '__main__':
    cwd = os.getcwd()
    file_path = os.path.join(cwd, 'data', 'test', 'tweets.txt')
    lines = read_txt_file(file_path)
    print(lines)
