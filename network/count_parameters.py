# Returns number of params
def count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
