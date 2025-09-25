"""
Kim et al (2025) equation on page6.

y=H_{c2}(T)

1. Kim, S. et al. Spin-orbit coupling induced enhancement of upper critical field in superconducting A15 single crystals. Journal of Alloys and Compounds 1037, 182350 (2025).

"""
#%%
from mpmath import nsum
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

def residual_exp_raw(x, t, t_c, slope, field, nu_range=(-1e4, 1e4)):
    """
    field should be not normalized! normalized = H/H_orb
    """
    return np.divide(1.0, t)-np.exp(rhs(field, slope, x[0], t, x[1], t_c, field_orb=np.array([1.0]*len(field)), nu_range=nu_range))

def residual_raw_slopefit(x, t, t_c, field, nu_range=(-1e4, 1e4)):
    return lhs(t)-rhs(field, x[2], x[0], t, x[1], t_c, field_orb=np.array([1.0]), nu_range=nu_range)

def residual_exp_raw_slopefit(x, t, t_c, field, nu_range=(-1e4, 1e4)):
    return np.divide(1.0, t)-np.exp(rhs(field, x[2], x[0], t, x[1], t_c, field_orb=np.array([1.0]), nu_range=nu_range))
    
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


def make_curl_bracket_for_nsum(field, slope, alpha, t, l_so, t_c, field_orb):
    def curl_bracket_for_nsum(j):
        return curl_bracket(j, field, slope, alpha, t, l_so, t_c, field_orb)
    return curl_bracket_for_nsum
    
def ___rhs(field, slope, alpha, t, l_so, t_c, field_orb, nu_range=None):
    """
    summation using mpmath.nsum
    """
    #if isinstance(t, np.ndarray):
    sums=[]
    for i in range(len(t)):
        new_curl_bracket=make_curl_bracket_for_nsum(field[i], slope[i], alpha, t[i], l_so, t_c[i], field_orb[i])
        sums.append(nsum(new_curl_bracket, [-np.inf, np.inf]))
    return np.array(sums)
    #else:
    #    new_curl_bracket=make_curl_bracket_for_nsum(i, field, slope, alpha, t, l_so, t_c, field_orb)
    #    return nsum(new_curl_bracket, [-np.inf, np.inf])

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

class SumInfinity():
    """
    $\\sum_i[func(i,x)]$ where i=range(-inf, inf).

    monitor if sum of current batch is small enough, stop.
    """
    def __init__(self, func, args,
                 batch_size=100,
                 batch_sum_ratio=1e-8,
                 tol_converge=1e-8,
                 tol_abs=1e-15,
                 max_iter=1e8,
                 residual_func="2nd_order"
                 ):
        self.func=func
        self.args=args
        self.batch_size=batch_size
        self.batch_sum_ratio=batch_sum_ratio
        self.tol_batch_converge=tol_converge
        self.tol_abs=tol_abs
        self.max_iter = max_iter
        self.max_batch=max_iter//(batch_size*2)
        self.residual_func="2nd_order"
        self.status=0

    def summation(self, center=0.0):
        # 
        batch_idx_pos=np.arange(center+1, self.max_iter+1, step=self.batch_size)
        batch_idx_neg=np.arange(center-1, -self.max_iter-1, step=-self.batch_size)
        sum=self.func(center, self.args)
        batch_decrease_rate=[]
        tot_iter=1.0
        self.status=1
        batch_sum0=None
        for batch_idx in range(self.max_batch):
            batch_sum=np.zeros_like(sum)
            for i in np.arange(batch_idx_pos[batch_idx], batch_idx_pos[batch_idx+1], step=1):
                batch_sum=batch_sum+self.func(i, self.args)
                tot_iter=tot_iter+1
            for i in np.arange(batch_idx_neg[batch_idx], batch_idx_neg[batch_idx+1], step=-1):
                batch_sum=batch_sum+self.func(i, self.args)
                tot_iter=tot_iter+1
            
            sum=sum+batch_sum
            batch_decrease_rate.append(np.sum(batch_sum, batch_sum0))
            if self.check_tol(batch_sum, batch_sum0, sum, tot_iter):
                #condition satisfied, stop
                return self.early_stop(sum, tot_iter, batch_decrease_rate, batch_sum)
            else:
                batch_sum0=batch_sum        
        if self.status==1:
            print("stop, max iteration condition reached")
            self.satus=2
        else:
            raise ValueError(self.status)
        
    def check_tol(self, batch_sum, batch_sum0, sum, tot_iter):
        # check 
        if self.tol_abs<np.max(batch_sum):
            self.status=3
            return True
        elif np.all(batch_sum<np.multiply(self.batch_sum_ratio*sum)):
            self.status=4
            print(f"(iter {tot_iter}) batch_sum {batch_sum} reached batch_sum_ratio.")
            return True
        elif np.max(np.abs(batch_sum-batch_sum0))<=self.tol_batch_converge:
            self.status=5
            print(f"(iter {tot_iter}) looks like batch-sum has been converged to constant values. batch_sum: {batch_sum}")
            return False
        else:
            return False
        
    def early_stop(self, sum, tot_iter, batch_decrease_rate, batch_sum):
        residual_sum = self.residual_sum(self, batch_decrease_rate, batch_sum)
        print(f"using {self.residual_func} approximation, {residual_sum/sum*100.0} (% of sum) is added")
        #if np.divide()

        return sum+residual_sum
    
#%%
if __name__=="__main__":
    from scipy.optimize import least_squares, shgo
    import matplotlib.pyplot as plt

    import pandas as pd
    import numpy as np
    from sklearn.metrics import r2_score
    #%%
    fig5df = pd.read_csv("fig5_flatten.csv")
    fig5df
    #%%
    t=fig5df['T/Tc'].to_numpy()
    field_norm=fig5df['H/H_orb'].to_numpy()
    #slope=fig5df['s1_slope'].to_numpy()
    slope=np.array([1.4]*len(t))

    #field_orb=fig5df['s1_H_orb'].to_numpy()
    field_orb=np.array([5.1]*len(t))

    #t_c=fig5df['s1_T_c'].to_numpy()
    t_c=np.array([5.28]*len(t))


    ms=(t, t_c, slope, field_norm, field_orb)
    #%%
    xs0=[0.6, 0.1]
    res_lsq=least_squares(residual, xs0, args=ms, bounds=((0.0,0.0), (5.0, 5.0)), verbose=1)
    #res_lsq=least_squares(whh.residual, xs0, args=ms, bounds=((0.0,0.0), (5.0, 5.0)), gtol=None, ftol=1e-12, verbose=2)
    print(f"res_lsq.x:{res_lsq.x}")