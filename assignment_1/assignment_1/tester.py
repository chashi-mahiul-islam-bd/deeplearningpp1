import torch
from tqdm import tqdm


def predict(conf):
    """
    this method contains the prediction code for the model
    :param conf:
    :return:
    """
    test_loss = 0
    accuracy = 0 
    count = 0
    criterion = conf["criterion"]
    model = conf["model"]
    device = conf["device"]
    model.eval()
    with torch.no_grad():
        for batch_idx, sample in enumerate(tqdm(conf["validation_dataloader"])):
            count += 1
            image = sample["image"].to(device)
            target = sample["target"].to(device)
            image_pred = model(image)
            loss = criterion(image_pred, target)
            test_loss = test_loss + (
                    (1 / (batch_idx + 1)) * (loss.data - test_loss)
            )
            ps = torch.exp(image_pred)
            top_p, top_class = ps.topk(1, dim=1)
            #print(top_p, top_class)
            equals = top_class == target.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor))
        
        print()
        print(f"Accuracy: {accuracy/count}")
    conf["test_loss"] = test_loss
    return conf