"""
Logger to send data to TensorBoard
implemented using
https://github.com/mateuszbuda/brain-segmentation-pytorch/blob/master/logger.py

"""
import os

from torch.utils.tensorboard import SummaryWriter


class Logger:
    """
    Logger Class that contains methods to send log data to TensorBoard
    """
    def __init__(self, log_dir):
        """

        :param log_dir: input for log directory
        """
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)
        self.writer = SummaryWriter(log_dir=log_dir)

    def scalar_summary(self, tag, value, step):
        """Add scalar data to summary.

        :param tag: Data identifier
        :param value: (float or string/blob name): Value to save
        :param step: Global step value to record
        :return:
        """
        self.writer.add_scalar(tag=tag, scalar_value=value, global_step=step)
        self.writer.flush()

    def model_graph(self, model, input_to_model):
        """Add graph data to summary
        This method is used to draw the model
        :param model:  Model to draw.
        :param input_to_model:A variable or a tuple of variables to be fed.
        :return:
        """
        self.writer.add_graph(model, input_to_model)
        self.writer.close()
