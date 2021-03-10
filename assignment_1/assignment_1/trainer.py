from pathlib import Path
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchsummary import summary
import datetime
from logger import Logger
from dataset import MnistDataset
from util_file import model_saving, model_training, model_validation, model_selection, check_load_model


def log_loss_summary(logger, loss, step, prefix=""):
    """

    :param logger:
    :param loss:
    :param step:
    :param prefix:
    :return:
    """
    logger.scalar_summary(prefix + "loss", loss, step)


def model_draw(logger, model, input_to_model):
    """

    :param logger:
    :param model:
    :param input_to_model:
    :return:
    """
    logger.model_graph(model, input_to_model)

def directory_creator(*argv):
    """

    :param argv: list directories to be created
    :return:
    """
    for arg in argv:
        if not Path(arg).is_dir():
            Path(arg).mkdir(parents=True)


def remove_files(*argv):
    """

    :param argv: list of files that has to be deleted
    :return:
    """
    for file in argv:
        if Path(file).is_file():
            Path(file).unlink()


def saving_model(conf):
    """
    This will save the trsining model
    :param conf: Configurator
    :return:
    """
    current_model_save_path, best_model_save_path = conf['current_model_save_path'], \
                                                          conf['best_model_save_path']
    timestamp = f"{datetime.datetime.now().date()}-{datetime.datetime.now().time()}"
    step = conf['step']

    # save best Gen train epoch model
    if conf["train_loss"] < conf['best_train_loss']:
        conf['best_train_loss'] = conf["train_loss"]
        model_saving(
            conf,
            f"{str(best_model_save_path)}/{timestamp}_best_model_{step}.pt",
        )

        # remove the previous best model files and set new one
        remove_files(conf["best_model"])
        best_model_list = list(best_model_save_path.rglob("*.pt"))
        conf["best_model"] = (
            best_model_list[-1] if best_model_list else ""
        )
        print(f"the best model path is {conf['best_model']}")

    # saving current model
    model_saving(
        conf,
        f"{str(current_model_save_path)}/{timestamp}_model_{step}.pt",
    )

    # remove old current model files and set new one
    remove_files(conf["current_model"])
    current_model_list = list(current_model_save_path.rglob("*.pt"))
    conf["current_model"] = (
        current_model_list[-1] if current_model_list else ""
    )
    print(f"the current model path is {conf['current_model']}")
    conf["current_epoch"] = conf['step']


def create_dataset(path, test=False):
    """

    Parameters
    ----------
    path: path to data directory
    test: specify whether the data is for training or testing
    Returns
    -------
    Loaded dataset

    """
    return MnistDataset(path, test=test)


def dataset_creation(conf):
    """
    This method will create the dataloader
    :param conf: Configurator
    :return:
    """
    parameters = {
        "batch_size": conf["batch_size"],
        "shuffle": True,
        "num_workers": 6,
    }

    use_cuda = torch.cuda.is_available()
    conf["device"] = torch.device("cuda:0" if use_cuda else "cpu")
    training_set = create_dataset(conf["train_path"], test=False)
    validation_set = create_dataset(conf["valid_path"], test=True)

    conf["training_dataloader"] = DataLoader(training_set, **parameters)
    conf["validation_dataloader"] = DataLoader(validation_set, **parameters)
    return conf


def train(conf):
    """
    This is the training method
    :param conf: Configurator
    :return:
    """

    conf = model_selection(conf)
    conf["model"] = conf["model"].to(conf["device"])
    conf = check_load_model(conf)
    conf["criterion"] = nn.NLLLoss()
    step = conf["current_epoch"]
    max_epoch = conf["max_epochs"]
    logger = Logger(conf["log_dir"])
    summary(conf["model"], (1, 16, 16), -1, "cuda")

    # draw the model
    dummy_input_tensor = torch.from_numpy(np.random.randn(1, 1, 16, 16)).float()
    dummy_input_tensor = dummy_input_tensor.to(conf["device"])
    model_draw(logger, conf["model"], dummy_input_tensor)
    del dummy_input_tensor

    conf["best_train_loss"] = float("inf")

    while step < max_epoch:
        conf = model_training(conf)
        log_loss_summary(logger, conf["train_loss"], step, "train")
        conf = model_validation(conf)
        log_loss_summary(logger, conf["valid_loss"], step, "valid_loss")

        print(f"Epoch No: {step}"
              f"training_loss: {conf['train_loss']}"
              f"training_accuracy: {conf['train_accuracy']}"
              f"validation_loss: {conf['valid_loss']}"
              f"validation_accuracy: {conf['valid_accuracy']}")
        step += 1
        conf["step"] = step
        saving_model(conf)


