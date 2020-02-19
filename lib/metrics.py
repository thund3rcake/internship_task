import numpy as np
import matplotlib.pyplot as plt
from lib.data import get_batch

EPS = 1e-10


def dice(true, pred):
    """Dice score.

    Parameters
    ----------
    true : np.ndarray, 2d
         Ground truth mask that consists of 2 unique values: 0 - denotes background,
         1 - denotes object.
    pred : np.ndarray, 2d
         Predicted mask that consists of 2 unique values: 0 - denotes background,
         1 - denotes object.

    Returns
    -------
    float from 0 to 1
        Dice score. The greater the value of dice score the better.

    Notes
    -----
    Masks should contains only 2 unique values, one of them must be 0, another value, that denotes
    object, could be different from 1 (for example 255).

    """
    true = true.astype(bool)
    pred = pred.astype(bool)

    intersection = (true & pred).sum()
    im_sum = true.sum() + pred.sum()

    return 2.0 * intersection / (im_sum + EPS)


def get_dice(true, pred):
    """Mean dice score.

    Parameters
    ----------
    true : list[np.ndarray] or np.ndarray
         List of ground truth masks or one mask that consists of 2 unique values:
         0 - denotes background, 1 - denotes object.
    pred : list[np.ndarray] or np.ndarray
         List of predicted masks or one mask that consists of 2 unique values:
         0 - denotes background, 1 - denotes object.

    Returns
    -------
    float from 0 to 1
        Dice score or mean dice score in case then lists of masks are passed.
        The greater the value of dice score the better.

    Notes
    -----
    Masks should contains only 2 unique values, one of them must be 0, another value, that denotes
    object, could be different from 1 (for example 255).
    
    """
    assert type(true) == type(pred), "Types of true and pred should be the same."
    if isinstance(true, list):
        return np.mean([dice(t, p) for t, p in zip(true, pred)])
    elif isinstance(true, np.ndarray):
        return dice(true, pred)
    else:
        raise TypeError("Wrong type.")


def get_avg_dice(net, device, path_to_val, val_annotations, threshold=0.35, val_length=200):
    """Returns average dice score, calculated on the validation dataset
    
    If the net contains batch normalisation, net.eval() should be called
    before calling this method.
    :param net: Net
        neural network to make predictions
    :param device: torch.device
        May be cuda:0 or cpu
    :param path_to_val: str
        path to validation dataset
    :param val_annotations dict
        annotations to validation images
    :param threshold: float
        hyperparameter of the neural network
    :param val_length: int
        number of images to take from validation dataset to compute average
    :return: float
        Average dice score, calculated on validation dataset of given length
    """""
    score = 0
    for i in range(val_length):
        x, y = get_batch([i], val_annotations, path_to_val, 'val')
        x = x[:, 0:3, :, :].to(device)
        preds = net.forward(x)
        preds = preds.data.cpu().squeeze()
        preds = (preds > threshold).float().numpy()
        y = y.squeeze().numpy()
        dice = get_dice(y, preds)
        score += dice

        score /= val_length
    return score


def plot_dices(net, device, val_path, val_annotations, min_t=0.1, max_t=0.6, val_length=200):
    """ Calculates average dices and shows it on a plot

    :param net: Net
        Neural network
    :param device: torch.device
        cuda:0 or cpu
    :param val_path: str
        Path to validation data
    :param val_annotations: dict
        Annotations
    :param min_t: float
        Minimal threshold
    :param max_t: float
        Maximal threshold
    :param val_length: int
        Number of validation images to use
    :return: thres, scores -- arrays of thresholds and dice scores
    """
    scores = []
    thres = np.arange(min_t, max_t, 0.05)
    for t in thres:
        t = t.round(3)
        dice = get_avg_dice(net, device, val_path, val_annotations, t, val_length)
        print("dice {} = {}".format(t, dice))
        scores.append(dice)
    plt.plot(thres, scores)
    plt.xlabel('Threshold')
    plt.ylabel('Dice score')
    plt.grid(True)
    plt.title('Dice score from threshold dependency')
    return thres, scores