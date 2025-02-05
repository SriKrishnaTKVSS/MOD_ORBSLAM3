import numpy as np
import matplotlib.pyplot as plt
from Differentiators import differentiators



def fourier_fft(_data):
    # say _data is of shape (m,n) with each column having one variable data
    num_var=_data.shape[1]

    _filtered_signal=np.zeros_like(_data)
    length=(_data.shape[0])
    for i in range(0,num_var-1):
        data=_data[:,i]
        #-----------------
        _data_hat=np.fft.fftshift(np.fft.fft(data))
        # cutoff=0.06
        _data_hat_mag=np.abs(_data_hat)
        _data_hat_mag_max=np.max(_data_hat_mag)
        # cutoff_freq_value=cutoff*_data_hat_mag_max
        # cutoff_freq_index=np.argmin(cutoff_freq_value)
        cutoff_freq_index = int(length/4) +1 

        #------------------------
        _data_hat[0:cutoff_freq_index] = 0
        _data_hat[length-cutoff_freq_index-1:]=0
        _data_hat_filtered=_data_hat

    _filtered_signal[:,i]=np.real(np.fft.ifft(np.fft.ifftshift(_data_hat_filtered)))
    return _filtered_signal


# # --------- Examples
# tf=2*np.pi
# N=256
# t=np.linspace(0,tf,N,endpoint=False)

# x1=np.sin(t)
# noise=0.05*np.sin(64*t)

# data=x1+noise

# _filtered_data=fourier_fft(data)

# # diff=differentiators()
# # diff_data=diff.robust_differentiator(data)
# # diff_filtered_data=diff.robust_differentiator(_filtered_data)


# plt.figure(1)
# plt.subplot(2,1,1)
# plt.plot(t,data,label='Orignal signal')
# plt.plot(t, _filtered_data,label='filtered_signal')
# plt.legend()

# # plt.subplot(2,1,2)
# # plt.plot(t,diff_data,label='Orignal signal diff')
# # plt.plot(t, diff_filtered_data,label='filtered_signal diff')

# plt.show()