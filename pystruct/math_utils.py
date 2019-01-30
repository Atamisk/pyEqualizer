def calc_beta(stress, strength):
    top = strength.mu - stress.mu
    bottom = (stress.sigma + strength.sigma)**0.5
    return top/bottom
