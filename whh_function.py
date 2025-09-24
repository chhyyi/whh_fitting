"""
Kim et al (2025) equation on page6.

y=H_{c2}(T)

1. Kim, S. et al. Spin-orbit coupling induced enhancement of upper critical field in superconducting A15 single crystals. Journal of Alloys and Compounds 1037, 182350 (2025).

"""
#%%
NU_MIN=-1e4
NU_MAX=1e4

from scipy.optimize import least_squares, shgo
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

def whh_lhs_substract_rhs(x, T, t_c, slope, field):
    """
    arguments
    xs: parameters to be optimized
    ms: measured/empirical/ whatever
    """
    # x[0]: alpha(maki param.)
    # x[1]: lambda_so (l_so)

    t = np.divide(T, t_c)
    kwargs={
        "field": field,
        "slope": slope,
        "alpha": x[0],
        "t": t,
        "l_so": x[1],
        "t_c": t_c
    }
    return lhs(t)-rhs(**kwargs)
    
def lhs(t):
    return np.log(np.divide(1.0, t))

def rhs(field, slope, alpha, t, l_so, t_c):
    """
    hbar is defined on the paper. not the conventional selection
    """
    sum=np.zeros(shape=field.shape)
    for i in np.arange(NU_MIN, NU_MAX+1,step=1.0, dtype=float):
        sum=sum+curl_bracket(i=i, field=field, slope=slope, alpha=alpha, t=t, l_so=l_so, t_c=t_c)
    return sum


def _rhs(field, slope, alpha, t, l_so, t_c):
    """
    hbar is defined on the paper. not the conventional selection
    """
    terms=[]
    for i in np.arange(NU_MIN, NU_MAX+1,step=1.0, dtype=float):
        terms.append(curl_bracket(i=i, field=field, slope=slope, alpha=alpha, t=t, l_so=l_so, t_c=t_c))
    return np.sum(terms, axis=0)
    
def curl_bracket(i, field,slope, alpha, t, l_so, t_c):
    """curl_bracket is `{}`
    """
    return np.divide(1.0,np.abs(2*i+1))-np.divide(1.0, square_bracket(i=i, field=field, slope=slope, alpha=alpha, t=t, l_so=l_so, t_c=t_c))

def square_bracket(i, field, slope, alpha, t, l_so, t_c):
    """
    not 1/[] but []
    """
    term1=np.abs(2*i+1)
    term2=np.divide(hbar(field,slope, t_c), t)
    term3_numerator=np.power((np.divide(alpha*hbar(field,slope, t_c), t)), 2)
    term3_denominator =np.abs(2*i+1)+np.divide(hbar(field,slope, t_c)+l_so, t)
    return term1+term2+np.divide(term3_numerator, term3_denominator)

def hbar(field,slope,t_c):
    return np.divide(field,t_c*slope)*(4/(np.pi**2))

#def orbital_upper_critical_field():
    #return -0.69*np.abs(SlopeVal)
def whh_lhs_substract_rhs_for_plot(x, T, t_c, slope, alpha, l_so):
    #y  = x[0] -> (field)
    t = np.divide(T, t_c)
    return lhs(t)-rhs(x[0], slope=slope, alpha=alpha, t=t, l_so=l_so, t_c=t_c)
#%% # ###################### TS1 fitting ##########################
measured = pd.read_csv("fig5_digitized_ts1_supple.csv")
#%%
measures=measured.to_numpy()
ms=(measured["T(K)"].to_numpy(), measured["T_c"].to_numpy(), measured["Slope(T/K)"].to_numpy(), measured["Hc2(T)"].to_numpy())
xs0=[0.6, 0.1]
res_lsq=least_squares(whh_lhs_substract_rhs, xs0, args=ms, bounds=((0.0,0.0), (5.0, 5.0)), gtol=None, ftol=1e-10, verbose=2)

print(f"res_lsq.x:{res_lsq.x}")

#%% plot alpha=0.6, l_so=1e-8 (parameters from page6) together.
len_x=15
x=np.linspace(0.1, 6.0, len_x)

alpha=0.6
l_so=0.0
plot_inps=pd.DataFrame(x, columns=["T"])
plot_inps["T_c"]=[5.4]*len_x
plot_inps['slope']=[1.29]*len_x
plot_inps["alpha"]=[alpha]*len_x
plot_inps['l_so']=[l_so]*len_x

y0=[5.0]
alpha0_6lso0 = [least_squares(whh_lhs_substract_rhs_for_plot, y0, args=inp.to_list(), bounds=(0.0, np.inf), gtol=None, ftol=1e-10).x for _, inp in plot_inps.iterrows()]

alpha=0.0
l_so=0.0
plot_inps=pd.DataFrame(x, columns=["T"])
plot_inps["T_c"]=[5.4]*len_x
plot_inps['slope']=[1.29]*len_x
plot_inps["alpha"]=[alpha]*len_x
plot_inps['l_so']=[l_so]*len_x

y0=[5.0]
alpha0lso0 = [least_squares(whh_lhs_substract_rhs_for_plot, y0, args=inp.to_list(), bounds=(0.0, np.inf)).x for _, inp in plot_inps.iterrows()]

alpha=res_lsq.x[0]
l_so=res_lsq.x[1]
plot_inps=pd.DataFrame(x, columns=["T"])
plot_inps["T_c"]=[5.4]*len_x
plot_inps['slope']=[1.29]*len_x
plot_inps["alpha"]=[alpha]*len_x
plot_inps['l_so']=[l_so]*len_x

y0=[5.0]
fit_params = [least_squares(whh_lhs_substract_rhs_for_plot, y0, args=inp.to_list(), bounds=(0.0, np.inf)).x for _, inp in plot_inps.iterrows()]
#%%
fig, ax = plt.subplots()
ax.plot(x, alpha0lso0, label="α=0.0, λ=0.0")
ax.plot(x, alpha0_6lso0, label="α=0.6, λ=0.0")
ax.plot(x, fit_params, label=f"α={res_lsq.x[0]:.02}, λ={res_lsq.x[1]:.02}")


ax.scatter(measured["T(K)"].to_numpy(), measured["Hc2(T)"].to_numpy(), label="TS1")

ax.set(xlabel='T(K)', ylabel='Hc2(T)')
ax.set_xlim(left=0.0)
ax.set_ylim(bottom=0.0)
ax.legend()
plt.show()

#%% ############## with normalized values, whole datatable ##############
# fitting
fig5df = pd.read_csv("fig5_flatten.csv")
t=fig5df['T/Tc'].to_numpy()
field_norm=fig5df['H/H_orb'].to_numpy()
slope=fig5df['s1_slope'].to_numpy()
field_orb=fig5df['s1_H_orb'].to_numpy()
t_c=fig5df['s1_T_c'].to_numpy()

ms=(t*t_c, t_c, slope, field_norm*field_orb)
#%%
xs0=[0.6, 0.1]
res_lsq=least_squares(whh_lhs_substract_rhs, xs0, args=ms, bounds=((0.0,0.0), (5.0, 5.0)), gtol=None, ftol=1e-16, verbose=2)

print(f"res_lsq.x:{res_lsq.x}")
#%% plot from equation;
len_x=15
x=np.linspace(0.1, 6.0, len_x)

alpha=0.6
l_so=0.0
slope=1.4
t_c=5.4
plot_inps=pd.DataFrame(x, columns=["T"])
plot_inps["T_c"]=[t_c]*len_x
plot_inps['slope']=[slope]*len_x
plot_inps["alpha"]=[alpha]*len_x
plot_inps['l_so']=[l_so]*len_x

y0=[5.0]
alpha0_6lso0 = [least_squares(whh_lhs_substract_rhs_for_plot, y0, args=inp.to_list(), bounds=(0.0, np.inf), gtol=None, ftol=1e-10).x for _, inp in plot_inps.iterrows()]

alpha=0.0
l_so=0.0
slope=1.4
t_c=5.4
plot_inps=pd.DataFrame(x, columns=["T"])
plot_inps["T_c"]=[t_c]*len_x
plot_inps['slope']=[slope]*len_x
plot_inps["alpha"]=[alpha]*len_x
plot_inps['l_so']=[l_so]*len_x

y0=[5.0]
alpha0lso0 = [least_squares(whh_lhs_substract_rhs_for_plot, y0, args=inp.to_list(), bounds=(0.0, np.inf)).x for _, inp in plot_inps.iterrows()]

alpha=res_lsq.x[0]
l_so=res_lsq.x[1]
plot_inps=pd.DataFrame(x, columns=["T"])
plot_inps["T_c"]=[t_c]*len_x
plot_inps['slope']=[slope]*len_x
plot_inps["alpha"]=[alpha]*len_x
plot_inps['l_so']=[l_so]*len_x

y0=[5.0]
fit_params = [least_squares(whh_lhs_substract_rhs_for_plot, y0, args=inp.to_list(), bounds=(0.0, np.inf)).x for _, inp in plot_inps.iterrows()]
#%%
fig, ax = plt.subplots()
h_c2_orb=5.1
ax.plot(np.divide(x, t_c), np.divide(alpha0lso0, h_c2_orb), label="α=0.0, λ=0.0")
ax.plot(np.divide(x, t_c), np.divide(alpha0_6lso0, h_c2_orb), label="α=0.6, λ=0.0")
ax.plot(np.divide(x, t_c), np.divide(fit_params, h_c2_orb), label=f"α={res_lsq.x[0]:.02}, λ={res_lsq.x[1]:.02}")

samples=['TS1', 'TS2', 'TS3', 'TS4']
sample_dfs=[fig5df[fig5df['sample']==sample] for sample in samples]
markers=['+','x','.','v']

axes_scatter=[ax.scatter(sample_df["T/Tc"].to_numpy(), sample_df["H/H_orb"].to_numpy(), label=sample, marker=marker) for sample_df, sample, marker in zip(sample_dfs, samples, markers)]

ax.set(xlabel='T(K)', ylabel='Hc2(T)')
ax.set_xlim(left=0.0)
ax.set_ylim(bottom=0.0)
ax.legend()
plt.show()

#%% #############Fitting with given values!!!#################
# where slope=1.4, H_c2_orb=5.1
# with T_c = 5.1/(0.69*slope) = 5.279503105590062
fig5df = pd.read_csv("fig5_flatten.csv")
t=fig5df['T/Tc'].to_numpy()
field_norm=fig5df['H/H_orb'].to_numpy()
slope=np.array([1.4]*len(t))
field_orb=np.array([5.1]*len(t))
t_c=np.array([5.279503105590062]*len(t))

ms=(t*t_c, t_c, slope, field_norm*field_orb)
#%%
xs0=[0.6, 0.1]
res_lsq_slope1_4=least_squares(whh_lhs_substract_rhs, xs0, args=ms, bounds=((0.0,0.0), (5.0, 5.0)), gtol=None, ftol=1e-16, verbose=2)

print(f"res_lsq.x:{res_lsq.x}")

#%%
#%% plot from equation; y0=[5.0]
len_x=15
x=np.linspace(0.1, 6.0, len_x)

h_c2_orb=5.1
y0=[5.0]
t_c=5.279503105590062
slope=1.4


alpha=0.6
l_so=0.0
plot_inps=pd.DataFrame(x, columns=["T"])
plot_inps["T_c"]=[t_c]*len_x
plot_inps['slope']=[slope]*len_x
plot_inps["alpha"]=[alpha]*len_x
plot_inps['l_so']=[l_so]*len_x

alpha0_6lso0 = [least_squares(whh_lhs_substract_rhs_for_plot, y0, args=inp.to_list(), bounds=(0.0, np.inf), gtol=None, ftol=1e-10).x for _, inp in plot_inps.iterrows()]

alpha=0.0
l_so=0.0
plot_inps=pd.DataFrame(x, columns=["T"])
plot_inps["T_c"]=[t_c]*len_x
plot_inps['slope']=[slope]*len_x
plot_inps["alpha"]=[alpha]*len_x
plot_inps['l_so']=[l_so]*len_x

alpha0lso0 = [least_squares(whh_lhs_substract_rhs_for_plot, y0, args=inp.to_list(), bounds=(0.0, np.inf)).x for _, inp in plot_inps.iterrows()]

alpha=res_lsq.x[0]
l_so=res_lsq.x[1]
plot_inps=pd.DataFrame(x, columns=["T"])
plot_inps["T_c"]=[t_c]*len_x
plot_inps['slope']=[slope]*len_x
plot_inps["alpha"]=[alpha]*len_x
plot_inps['l_so']=[l_so]*len_x

fit_params = [least_squares(whh_lhs_substract_rhs_for_plot, y0, args=inp.to_list(), bounds=(0.0, np.inf)).x for _, inp in plot_inps.iterrows()]

fig, ax = plt.subplots()
ax.plot(np.divide(x, t_c), np.divide(alpha0lso0, h_c2_orb), label="α=0.0, λ=0.0")
ax.plot(np.divide(x, t_c), np.divide(alpha0_6lso0, h_c2_orb), label="α=0.6, λ=0.0")
ax.plot(np.divide(x, t_c), np.divide(fit_params, h_c2_orb), label=f"α={res_lsq.x[0]:.02}, λ={res_lsq.x[1]:.02}")

samples=['TS1', 'TS2', 'TS3', 'TS4']
sample_dfs=[fig5df[fig5df['sample']==sample] for sample in samples]
markers=['+','x','.','v']

axes_scatter=[ax.scatter(sample_df["T/Tc"].to_numpy(), sample_df["H/H_orb"].to_numpy(), label=sample, marker=marker) for sample_df, sample, marker in zip(sample_dfs, samples, markers)]

ax.set(xlabel='T(K)', ylabel='Hc2(T)')
ax.set_xlim(left=0.0)
ax.set_ylim(bottom=0.0)
ax.legend()
plt.show()

#%% plot from equation; y0=[0.6]
len_x=15
x=np.linspace(0.1, 6.0, len_x)

h_c2_orb=5.1
y0=[0.6]
t_c=5.279503105590062

alpha=0.6
l_so=0.0
slope=1.4
plot_inps=pd.DataFrame(x, columns=["T"])
plot_inps["T_c"]=[t_c]*len_x
plot_inps['slope']=[slope]*len_x
plot_inps["alpha"]=[alpha]*len_x
plot_inps['l_so']=[l_so]*len_x

alpha0_6lso0 = [least_squares(whh_lhs_substract_rhs_for_plot, y0, args=inp.to_list(), bounds=(0.0, np.inf), gtol=None, ftol=1e-10).x for _, inp in plot_inps.iterrows()]

alpha=0.0
l_so=0.0
plot_inps=pd.DataFrame(x, columns=["T"])
plot_inps["T_c"]=[t_c]*len_x
plot_inps['slope']=[slope]*len_x
plot_inps["alpha"]=[alpha]*len_x
plot_inps['l_so']=[l_so]*len_x

alpha0lso0 = [least_squares(whh_lhs_substract_rhs_for_plot, y0, args=inp.to_list(), bounds=(0.0, np.inf)).x for _, inp in plot_inps.iterrows()]

alpha=res_lsq.x[0]
l_so=res_lsq.x[1]
plot_inps=pd.DataFrame(x, columns=["T"])
plot_inps["T_c"]=[t_c]*len_x
plot_inps['slope']=[slope]*len_x
plot_inps["alpha"]=[alpha]*len_x
plot_inps['l_so']=[l_so]*len_x

fit_params = [least_squares(whh_lhs_substract_rhs_for_plot, y0, args=inp.to_list(), bounds=(0.0, np.inf)).x for _, inp in plot_inps.iterrows()]
#%%
fig, ax = plt.subplots()
ax.plot(np.divide(x, t_c), np.divide(alpha0lso0, h_c2_orb), label="α=0.0, λ=0.0")
ax.plot(np.divide(x, t_c), np.divide(alpha0_6lso0, h_c2_orb), label="α=0.6, λ=0.0")
ax.plot(np.divide(x, t_c), np.divide(fit_params, h_c2_orb), label=f"α={res_lsq.x[0]:.02}, λ={res_lsq.x[1]:.02}")

samples=['TS1', 'TS2', 'TS3', 'TS4']
sample_dfs=[fig5df[fig5df['sample']==sample] for sample in samples]
markers=['+','x','.','v']

axes_scatter=[ax.scatter(sample_df["T/Tc"].to_numpy(), sample_df["H/H_orb"].to_numpy(), label=sample, marker=marker) for sample_df, sample, marker in zip(sample_dfs, samples, markers)]

ax.set(xlabel='T(K)', ylabel='Hc2(T)')
ax.set_xlim(left=0.0)
ax.set_ylim(bottom=0.0)
ax.legend()
plt.figtext(0.5, -0.05, "(fit/plot) h_c2_orb(0)(T)=5.1, T_c(K)=5.2795, slope=1.4", wrap=True, horizontalalignment='center', fontsize=12)
plt.show()