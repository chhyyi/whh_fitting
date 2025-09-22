"""
Kim et al (2025) equation on page6.

1. Kim, S. et al. Spin-orbit coupling induced enhancement of upper critical field in superconducting A15 single crystals. Journal of Alloys and Compounds 1037, 182350 (2025).

"""
import numpy as np
def lhs(t, tc):
    return np.log(t/tc)

def rhs(habr, alpha, t, lambda_so, i_min=-100, i_max=100):
    """
    hbar is defined on the paper. not the conventional selection
    """
    parts=[]
    for i in range(i_min, i_max):
        parts.append(curl_bracket(i, hbar, alpha, t, lambda_so))
    
def curl_bracket(i, hbar, alpha, t, lambda_so):
    """curl_bracket is `{}`
    """
    return 1/np.abs(2*i)-1/square_bracket(i, hbar, t, lambda_so)

def square_bracket():
    pass

def hbar():
    pass

