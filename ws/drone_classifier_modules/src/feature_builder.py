import numpy as np
import pywt
from scipy.stats import kurtosis


def compute_dwt_statistics(timeseries_matrix, waveletname='db4', level=3):
    """
    Extracts DWT-based statistical features from time-series data.
    """
    flight_features = []
    min_len = pywt.Wavelet(waveletname).dec_len * (2 ** level)
    
    if timeseries_matrix.shape[0] < min_len:
        pad_width = min_len - timeseries_matrix.shape[0]
        timeseries_matrix = np.pad(timeseries_matrix, ((0, pad_width), (0, 0)), mode='edge')
    
    for i in range(timeseries_matrix.shape[1]):
        signal = timeseries_matrix[:, i]
        coeffs = pywt.wavedec(signal, waveletname, level=level)
        
        coeffs_to_use = coeffs[:3] 
        for coeff in coeffs_to_use:
            mean_val = np.mean(coeff)
            std_val = np.std(coeff)
            energy_val = np.sum(np.square(coeff)) 
            max_val = np.max(coeff)
            min_val = np.min(coeff)
            kurt_val = kurtosis(coeff)
            peak_loc_max = np.argmax(coeff) / len(coeff)
            peak_loc_min = np.argmin(coeff) / len(coeff)
            
            flight_features.extend([
                mean_val, std_val, energy_val, max_val, min_val, 
                kurt_val, peak_loc_max, peak_loc_min
            ])
            
    return np.array(flight_features)

def pad_sequences(X_list):
    """
    Pads variable-length sequences with zeros to match the maximum length.
    Returns a 3D numpy array: (num_samples, max_length, num_features).
    """
    max_len = max(len(x) for x in X_list)
    num_features = X_list[0].shape[1]
    X_padded = np.zeros((len(X_list), max_len, num_features))
    
    for i, x in enumerate(X_list):
        X_padded[i, :len(x), :] = x
        
    return X_padded
