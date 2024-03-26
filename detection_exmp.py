from scipy.io import loadmat
import scipy
from detection import get_events
import os


import pyabf

path = r"C:\Users\ayber\Desktop\Checken\LT19_3MKCl_1nMPOC_800mV_002.abf"
abf = pyabf.ABF(path)
signal = abf.data[0]
signal = signal[0:10000000000]




samplerate = 40_000_000                #samplerate of signal

rough_detec_params = {
    "s": 5,                         #threshhold of detecting start of event
    "e": 0,                         #threshhold of detection end of event (after start was detected)
    # "dt_exact": 50,               #number of points used to calculate local std/mean (only use when using rd_algo : exact, but generally not neccessary)
    # "rd_algo": "exact",           #if left out, use a recursive lowpass filter for thresh hold, in general leave this out
    # "lag": 1,                     #only necessary when rd_algo : exact is used. Leave it at 1 or 2, not that important
    # "max_event_length": 10e-5,    #maximum duration for a signal to be considered an event and not pore clogging for example (in SI units)
}

fit_method = "c_pelt"               #method to perform level fitting. Can be c_pelt, c_dynamic, cusum, pelt, dynamic
                                    #(difference between c_pelt and pelt are, that with c_pelt you can only use "model": "l2", same with c_dynamic)
                                    #c_pelt is for detecting unknown amounts of levels. Generally leave it at c_pelt or pelt.

fit_params = {
    # "delta": 0.2,                 #parameter for cusum
    # "hbook": 1,                   #parameter for cusum
    # "sigma": 0.0387,              #parameter for cusum
    # "dt_baseline": 25,            #controlls how much of the baseline is given to perform level fitting (need some amout of baseline to see where event starts)
    "fit_event_thresh": 5,         #minimum length of entire event (in number of samples)
    "fit_level_thresh": 3,          #minimum length of one level (in number of samples)
    "pen":  100, #"BIC",                  #sensitivity parameter for pelt/c_pelt. Either pass float (the lower the more levels are detected) or leave it as
                                    #"BIC" or "AIC" for unsupervised level fitting. In paper more information on this.
    "model": "l2",                  #controls what type of changepoints are detected. In Documentation there is more information on this
    # "nr_ct": 5,                   #number of set changepoints if you use fit_method: dynamic/c_dynamic
}


show = False                        #if set to true, you can see how the algorithm detects the levels

save_folder = r"C:\Users\ayber\Desktop\pelt_test\json_save"                                     #path to the folder where you want to save output
file_name = "result"                                                                            #name of the output file
get_events(signal, samplerate, rough_detec_params=rough_detec_params, fit_method=fit_method,
           fit_params=fit_params, show=show, folder_path=save_folder, filename=file_name)






########### information for onput file ############
"""
output file is given as a json file. Written inside are the parameters you set to let the algorithm run. The results are 
saved in a list called "events". Each element of the list represents one event. Inside one element there is:
"signal_w_baseline": the entire signal that was used for the level fitting (includes samples from the baseline)
"start_end_in_sig": the datapoints, where the actual event starts end ends in "signal_w_baseline"
"local_baseline": the local baseline before the event started.
"level_info": information about the levels. This is a list, where one element represents one found level inside the given event.
              One element consists of two values:
              The first value is the drop relative to the baseline, e.g -0.5 would mean that the level is 0.5 below the baseline 
              Second value is how long the event lasted in number of samples
"mean": average current drop, also relative to baseline
"start_end_in_raw": the datapoints, where the event starts in the raw signal (not just the provided signal that is to be fitted)
"std": standard deviation inside the event
"change_times": change times detected by algorithm. The segment between two changet times is one level. First changetime is where event starts and
                last change time is where event ends in the provided "signal_w_baseline"
"height": the height value
"dwell": the dwell time of the event

If you want to get some other features out of the event, you need to adjust the "Event" class inside the "detection.py" file
"""







