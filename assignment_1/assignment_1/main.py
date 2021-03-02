import toml
from trainer import dataset_creation, train


def process(conf):
    """
    
    :param conf: Configurator 
    :return: 
    """
    conf = dataset_creation(conf)
    train(conf)

if __name__=="__main__":
    conf = toml.load("../assignment_1/assignment_1/mainconfig.toml")
    process(conf["dataset_training"])

