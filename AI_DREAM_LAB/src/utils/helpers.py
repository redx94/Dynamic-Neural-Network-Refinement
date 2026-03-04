
def compute_accuracy(predictions, labels):
    correct = (predictions == labels).sum().item()
    total = labels.size(0)
    return correct / total

def log_metrics(epoch, metrics, logger):
    logger.log({f"Epoch_{epoch}": metrics})
    