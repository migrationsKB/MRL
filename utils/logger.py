import logging


def logger(output_file: str) -> None:
    """
    Logger for the program
    :param output_file:
    :return:
    """
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s',
                        filename=f'output/logs/{output_file}.log',
                        filemode='w')
    logging.debug("Debug message")
    logging.info("Informative message")
    logging.error("Error message")