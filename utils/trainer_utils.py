import random
import torch


def track_value(logger, epoch, step, loss, y_pred, y_gt):
    # logger.info(
    #     "\n\nEpoch: {}, Step: {}, MSE loss = {:.5f}".format(
    #         epoch + 1, step, loss.item()
    #     )
    # )
    print_idx = random.sample(range(len(y_gt)), 1)[0]
    logger.info("y_pred: {}, y_gt: {}".format(y_pred.shape, y_gt.shape))
    logger.info("y_pred[{}]: {}\n".format(print_idx, y_pred[print_idx]))
    logger.info("y_gt[{}]: {}\n".format(print_idx, y_gt[print_idx]))


def check_nan(logger, loss, y_pred, y_gt):
    if torch.any(torch.isnan(loss)):
        logger.info("out has nan: ", torch.any(torch.isnan(y_pred)))
        logger.info("y_gt has nan: ", torch.any(torch.isnan(y_gt)))
        logger.info("out: ", y_pred)
        logger.info("y_gt: ", y_gt)
        logger.info("loss = {:.4f}\n".format(loss.item()))
        exit()
