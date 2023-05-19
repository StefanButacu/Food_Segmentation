import torch


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    print("=>saving checkpoint to",  filename)
    torch.save(state, filename)



def load_checkpoint(checkpoint, model):
    model.load_state_dict(checkpoint["state_dict"])