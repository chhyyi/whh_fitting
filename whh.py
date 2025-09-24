"""
Kim et al (2025) equation on page6.

y=H_{c2}(T)

1. Kim, S. et al. Spin-orbit coupling induced enhancement of upper critical field in superconducting A15 single crystals. Journal of Alloys and Compounds 1037, 182350 (2025).

"""


import numpy as np
def residual(x, t, t_c, slope, field_norm, field_orb, nu_range=(-1e4, 1e4)):
    """
    arguments
    xs: parameters to be optimized
    ms: measured/empirical/ whatever
    """
    # x[0]: alpha(maki param.)
    # x[1]: lambda_so (l_so)

    return lhs(t)-rhs(field_norm, slope, x[0], t, x[1], t_c, field_orb, nu_range=nu_range)

def residual_exp(x, t, t_c, slope, field_norm, field_orb, nu_range=(-1e4, 1e4)):
    return np.divide(1.0, t)-np.exp(rhs(field_norm, slope, x[0], t, x[1], t_c, field_orb, nu_range=nu_range))
    
def residual_for_plot(x, t, t_c, slope, alpha, l_so, field_orb, nu_range=(-1e4, 1e4)):
    return lhs(t)-rhs(x[0], slope, alpha, t, l_so, t_c, field_orb, nu_range=nu_range)

def residual_exp_for_plot(x, t, t_c, slope, alpha, l_so, field_orb, nu_range=(-1e4, 1e4)):
    return np.divide(1.0,t)-np.exp(rhs(x[0], slope, alpha, t, l_so, t_c, field_orb, nu_range=nu_range))

def lhs(t):
    return np.log(np.divide(1.0, t))

def _rhs(field, slope, alpha, t, l_so, t_c, field_orb, nu_range):
    """
    hbar is defined on the paper. not the conventional selection
    """
    sum=np.zeros(shape=field.shape)
    for i in np.arange(nu_range[0], nu_range[1]+1,step=1.0):
        sum=sum+curl_bracket(i, field, slope, alpha, t, l_so, t_c, field_orb)
    return sum


def rhs(field, slope, alpha, t, l_so, t_c, field_orb, nu_range=(-1e3,1e3)):
    """
    hbar is defined on the paper. not the conventional selection
    """
    terms=[]
    for i in np.arange(nu_range[0], nu_range[1]+1,step=1.0):
        terms.append(curl_bracket(i, field, slope, alpha, t, l_so, t_c, field_orb))
    return np.sum(terms, axis=0)
    
def curl_bracket(i, field,slope, alpha, t, l_so, t_c, field_orb):
    """curl_bracket is `{}`
    """
    return np.divide(1.0,np.abs(2*i+1))-np.divide(1.0, square_bracket(i, field, slope, alpha, t, l_so, t_c, field_orb))

def square_bracket(i, field, slope, alpha, t, l_so, t_c,field_orb):
    """
    not 1/[] but []
    """
    term1=np.abs(2*i+1)
    term2=np.divide(hbar(field,slope, t_c,field_orb), t)
    term3_numerator=np.power((np.divide(alpha*hbar(field,slope, t_c,field_orb), t)), 2)
    term3_denominator =np.abs(2*i+1)+np.divide(hbar(field,slope, t_c,field_orb)+l_so, t)
    return term1+term2+np.divide(term3_numerator, term3_denominator)

def hbar(field,slope,t_c,field_orb):
    return np.divide(field*field_orb,t_c*slope)*(4/(np.pi**2))

