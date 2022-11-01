import logging


def create_logger(log_name):
    # filter
    class fileFilter(logging.Filter):
        def filter(self, record):
            return (not record.getMessage().startswith("Added")) and (
                not record.getMessage().startswith("Rank ")
            )

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        format="%(asctime)s| %(message)s",
        datefmt="%m-%d:%H:%M:%S",
        level=logging.INFO,
        handlers=[logging.FileHandler(log_name, mode="a"), logging.StreamHandler()],
    )

    for handler in logging.root.handlers:
        handler.addFilter(fileFilter())


if __name__ == '__main__':
    pass

