import os
import yaml


def load_academic_research_bearer(input_dir, api_name):
    """
    Load authentication information
    :param input_dir:
    :param api_name:
    :return:
    """
    credentials = yaml.load(open(os.path.join(input_dir, 'crawler', 'config', 'credentials.yaml')), yaml.FullLoader)
    bearer_token = credentials[api_name]['bearer_token']
    return bearer_token
