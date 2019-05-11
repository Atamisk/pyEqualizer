def calc_beta(stress, strength):
    top = strength.mu - stress.mu
    bottom = (stress.sigma**2 + strength.sigma**2)**0.5
    return top/bottom
