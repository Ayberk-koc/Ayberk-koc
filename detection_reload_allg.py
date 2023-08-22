import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import matplotlib.pyplot as plt
import ruptures as rpt
import pandas as pd
# from Nanopore_tools.Functions import CUSUM
import scipy
from scipy import signal as sig
import time as tm
from ruptures.exceptions import BadSegmentationParameters
import json

class NotFittable(Exception):
    pass


def CUSUM(input, delta, h, verbose=False):
    """
    Function used in the detection of abrupt changes in mean current; optimal for Gaussian signals.
    CUSUM is based on the cummulative sum algorithm.
    This function will define new start and end points more precisely than just
    the RecursiveLowPassFast and will fit levels inside the TransolocationEvent objects.

    Parameters
    ----------
    input : numpy array
        Input signal.
    delta : float
        Most likely jump to be detected in the signal.
    h : float
        Threshold for the detection test.

    Returns
    -------
    mc : the piecewise constant segmented signal
    kd : a list of float detection times (in samples)
    krmv : a list of float estimated change times (in samples).
    """

    # initialization
    Nd = k0 = 0
    kd = []
    krmv = []
    k = 1
    l = len(input)
    m = np.zeros(l)
    m[k0] = input[k0]
    v = np.zeros(l)
    sp = np.zeros(l)
    Sp = np.zeros(l)
    gp = np.zeros(l)
    sn = np.zeros(l)
    Sn = np.zeros(l)
    gn = np.zeros(l)

    while k < l:
        m[k] = np.mean(input[k0:k + 1])
        v[k] = np.var(input[k0:k + 1])

        sp[k] = delta / v[k] * (input[k] - m[k] - delta / 2)
        sn[k] = -delta / v[k] * (input[k] - m[k] + delta / 2)

        Sp[k] = Sp[k - 1] + sp[k]
        Sn[k] = Sn[k - 1] + sn[k]

        gp[k] = np.max([gp[k - 1] + sp[k], 0])
        gn[k] = np.max([gn[k - 1] + sn[k], 0])

        if gp[k] > h or gn[k] > h:
            kd.append(k)
            if gp[k] > h:
                kmin = np.argmin(Sp[k0:k + 1])
                krmv.append(kmin + k0)
            else:
                kmin = np.argmin(Sn[k0:k + 1])
                krmv.append(kmin + k0)

            # Re-initialize
            k0 = k
            m[k0] = input[k0]
            v[k0] = sp[k0] = Sp[k0] = gp[k0] = sn[k0] = Sn[k0] = gn[k0] = 0

            Nd = Nd + 1
        k += 1
    if verbose:
        print('delta:' + str(delta))
        print('h:' + str(h))
        print('Nd: ' + str(Nd))
        print('krmv: ' + str(krmv))

    if Nd == 0:
        mc = np.mean(input) * np.ones(k)
    elif Nd == 1:
        mc = np.append(m[krmv[0]] * np.ones(krmv[0]), m[k - 1] * np.ones(k - krmv[0]))
    else:
        mc = m[krmv[0]] * np.ones(krmv[0])
        for ii in range(1, Nd):
            mc = np.append(mc, m[krmv[ii]] * np.ones(krmv[ii] - krmv[ii - 1]))
        mc = np.append(mc, m[k - 1] * np.ones(k - krmv[Nd - 1]))
    return (mc, kd, krmv)

def ct_pelt(signal, model="normal", pen=5):
    """
    detection für die changetimes.
    Parameters
    ----------
    signal      array: gebe hier signale mit bisschen baseline ein
    model       str: cost function
    pen         float: penalty value, constolls tradeoff for between goodness of fit and ct detected
                       the lower, the more sensitive

    Returns
    -------

    """
    algo = rpt.Pelt(model=model).fit(signal)
    ct = algo.predict(pen=pen)    #hier sind die changetimes. Der start ist nicht dabei. Dann die change times. d.h ct[0] ist start des events
    return ct                     #doch es ist ct[-2] ende des events, weil ende des signal auch eine ct ist. Die changetimes dazwischen sind dann für die lvls


# def cusum(signal, delta, h):
#     lvls, det_times, ct = CUSUM(signal, delta, h)
#     ##
#     return lvls, det_times, ct


def rough_detect_2(signal, samplerate, dt=100, s=5, e=0, max_event_length=5e-1):

    signal = np.ravel(signal)

    init_base = np.ones(dt) * np.mean(signal[:dt])
    init_var = np.ones(dt) * np.var(signal[:dt])

    window = sliding_window_view(signal, window_shape=(dt,))       #hat 99 weniger als signal!
    #print(window.shape)
    ml_post_init = np.mean(window, axis=1)
    #print(local_baseline.shape)
    vl_post_init = np.var(window, axis=1)


    ml = np.concatenate((init_base, ml_post_init))
    vl = np.concatenate((init_var, vl_post_init))


    sl = ml - s * np.sqrt(vl)       #das ist array


    time = np.linspace(0, len(signal)/samplerate, len(signal))
    plt.plot(time, signal, color="red")
    plt.plot(time, np.delete(sl, -1), color="green")
    plt.show()
    exit()



    Ni = len(signal)
    points = np.array(np.where(signal <= np.delete(sl, -1))[0])        #sl
    to_pop = np.array([], dtype=int)  # in den np.array habe ich auch ein dtype rein gemacht
    for i in range(1, len(points)):
        if points[i] - points[i - 1] == 1:
            to_pop = np.append(to_pop, i)
    points = np.unique(np.delete(points, to_pop))
    RoughEventLocations = []
    NumberOfEvents = 0

    print(len(points))  #hier gemacht, später del
    print(points)   #hier gemacht, später del
    exit()  #hier gemacht, später del

    for i in points:
        if NumberOfEvents != 0:
            if i >= RoughEventLocations[NumberOfEvents - 1][0] and i <= RoughEventLocations[NumberOfEvents - 1][1]:
                continue
        NumberOfEvents += 1
        start = i
        El = ml[i] + e * np.sqrt(vl[i])
        Mm = ml[i]
        Vv = vl[i]
        duration = 0
        endp = start
        if (endp + 1) < len(signal):
            while signal[endp + 1] < El and endp < (Ni - 2):  # and duration < coeff['maxEventLength']*samplerate:
                duration += 1
                endp += 1
        if duration >= max_event_length * samplerate or endp > (
                Ni - 10):  # or duration <= coeff['minEventLength'] * samplerate:
            NumberOfEvents -= 1
            continue
        else:
            k = start
            while signal[k] < Mm and k > 1:
                k -= 1
            start = k - 1
            k2 = i + 1
            # while signal[k2] > Mm:
            #    k2 -= 1
            # endp = k2
            if start < 0:
                start = 0
            RoughEventLocations.append((start, endp, np.sqrt(vl[start]), ml[start]))     #ml/vl ist local baseline/local variance bei start des events
                                                                                         #nehme also die standard abweichung
    return np.array(RoughEventLocations)  # , ml, vl, sl




def RecursiveLowPassFast(signal, **rough_detec): #s=5, e=0, a=0.999, max_event_length=5e-1
    """
    Function used to find roughly where events are in a noisy signal using a first order recursive
    low pass filter defined as :
        u[k] = a*u[k-1]+(1-a)*i[k]
        with u the mean value at sample k, i the input signal and a < 1, a parameter.
    """
    s = rough_detec.get("s", 5)
    e = rough_detec.get("e", 0)
    a = rough_detec.get("a", 0.999)
    max_event_length = rough_detec.get("max_event_length", 5e-1)
    samplerate = rough_detec.get("samplerate")



    signal = np.ravel(signal)

    padlen = np.uint64(samplerate)
    prepadded = np.ones(padlen) * np.mean(signal[0:1000])
    signaltofilter = np.concatenate((prepadded, signal))

    mltemp = scipy.signal.lfilter([1 - a, 0], [1, -a], signaltofilter)
    vltemp = scipy.signal.lfilter([1 - a, 0], [1, -a], np.square(signaltofilter - mltemp))

    ml = np.delete(mltemp, np.arange(padlen, dtype=int))  # in dem np.arange habe ich das type=int hinzugefügt
    vl = np.delete(vltemp, np.arange(padlen, dtype=int))  # in dem np.arange habe ich das type=int hinzugefügt


    sl = ml - s * np.sqrt(vl)       #das ist array


    # time = np.linspace(0, len(signal)/samplerate, len(signal))
    # plt.plot(time, signal, color="red")
    # plt.plot(time, sl, color="green")
    # plt.show()

    Ni = len(signal)
    points = np.array(np.where(signal <= sl)[0])
    to_pop = np.array([], dtype=int)  # in den np.array habe ich auch ein dtype rein gemacht
    for i in range(1, len(points)):
        if points[i] - points[i - 1] == 1:
            to_pop = np.append(to_pop, i)
    points = np.unique(np.delete(points, to_pop))
    RoughEventLocations = []
    NumberOfEvents = 0

    for i in points:
        if NumberOfEvents != 0:
            if i >= RoughEventLocations[NumberOfEvents - 1][0] and i <= RoughEventLocations[NumberOfEvents - 1][1]:
                continue
        NumberOfEvents += 1
        start = i
        El = ml[i] + e * np.sqrt(vl[i])
        Mm = ml[i]
        Vv = vl[i]
        duration = 0
        endp = start
        if (endp + 1) < len(signal):
            while signal[endp + 1] < El and endp < (Ni - 2):  # and duration < coeff['maxEventLength']*samplerate:
                duration += 1
                endp += 1
        if duration >= max_event_length * samplerate or endp > (
                Ni - 10):  # or duration <= coeff['minEventLength'] * samplerate:
            NumberOfEvents -= 1
            continue
        else:
            k = start
            while signal[k] < Mm and k > 1:
                k -= 1
            start = k - 1
            k2 = i + 1
            # while signal[k2] > Mm:
            #    k2 -= 1
            # endp = k2
            if start < 0:
                start = 0
            RoughEventLocations.append((start, endp, np.sqrt(vl[start]), ml[start]))     #ml/vl ist local baseline/local variance bei start des events
                                                                                         #nehme also die standard abweichung
    return np.array(RoughEventLocations)  # , ml, vl, sl





def find_rough_event_loc(signal, **rough_detec):       #dt=100, s=5, lag=2

    dt = rough_detec.get("dt", 100)
    s = rough_detec.get("s", 5)
    lag = rough_detec.get("lag", 0)

    event_start_indices = []
    event_end_indices = []
    event_local_std = []
    event_local_baseline = []

    if isinstance(signal, np.ndarray):
        signal = np.ravel(signal)


    window = sliding_window_view(signal, window_shape=(dt,))
    local_baseline = np.mean(window, axis=1)
    local_std = np.std(window, axis=1)


    #event_start_thresh = []         #ZUM PLOTTEN

    event_start_detected = False
    for i in range(dt, len(signal)):

        if not event_start_detected:
            #local_baseline = np.mean(signal[i-dt:i])
            #local_std = np.std(signal[i-dt:i])                 #das brauchst du für die cusums! deshalb mache das auch rein in die indices!
            event_start_threshold = local_baseline[i-dt-1] - s * local_std[i-dt-1-lag]    #meistens reicht lag=3 aus!!
            event_end_threshold = local_baseline[i-dt-1] # - e * local_std[i-dt-1-lag]    #könnte auch in baseline einen lag machen

        #event_start_thresh.append(event_start_threshold)        #ZUM PLOTTEN

        if not event_start_detected and signal[i] < event_start_threshold and signal[i - 1] >= event_start_threshold:
            event_start_indices.append(i)
            event_local_baseline.append(local_baseline[i-dt])
            event_local_std.append(local_std[i-dt])        #hier damit ich für cusum local std habe
            event_start_detected = True

        if event_start_detected:
            if signal[i] > event_end_threshold and signal[i - 1] <= event_end_threshold:
                event_end_indices.append(i)
                event_start_detected = False

    #hier könnte memory frei machen!

    ### hier die event indices
    event_rough_infos = []
    for i in range(len(event_start_indices)):
        try:
            if (event_end_indices[i] - event_start_indices[i]) < 10: continue       #hier um die impulse weg zu machen
            event_rough_infos.append((event_start_indices[i], event_end_indices[i], event_local_std[i], event_local_baseline[i]))
        except IndexError:
            continue

    # time = np.linspace(0, len(signal[dt:])/1_000, len(signal[dt:]))        #ZUM PLOTTEN
    # plt.plot(time, signal[dt:], label='Signal with Event')
    # plt.plot(time, event_start_thresh, color="red", linewidth=1)
    # plt.xlabel('Time')
    # plt.ylabel('Current (nA)')
    # plt.title('Signal with Baseline, Event, and Two Levels with Noise')
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    return event_rough_infos




class Events:
    show = False

    def __init__(self, ev_infos, samplerate, samp_dwell_thresh, signal, dt_baseline=50,
                 fit_method="pelt", **fit_kwargs):    #hier noch überlegen was mit den fit_kwargs machen soll
        # self.event_indices = [(ev[0], ev[1]) for ev in ev_infos]
        # self.local_baseline_stds = [ev[2] for ev in ev_infos]
        self.dt_baseline = dt_baseline
        self.ev_infos = ev_infos    #tuple mit (ev_start_ind, ev_end_ind, local_std, local_baseline)
        self.samplerate = samplerate
        self.samp_dwell_thresh = samp_dwell_thresh
        self.signal = signal
        self.fit_method = fit_method
        self.fit_kwargs = fit_kwargs
        self.events = self.create_events()

    def __iter__(self):
        self.i = 0
        return self

    def __next__(self):
        if self.i >= len(self.events):
            raise StopIteration
        value = self.events[self.i]
        self.i += 1
        return value

    def create_events(self):
        events = []
        EventReload.samplerate = self.samplerate
        EventReload.samp_dwell_thresh = self.samp_dwell_thresh
        EventReload.fit_time = self.fit_kwargs.get("fit_time")
        for ev_info in self.ev_infos:
            ev_start = int(ev_info[0])   #start und endpunkt index
            ev_end = int(ev_info[1])
            local_std = ev_info[2]
            local_baseline = ev_info[3]
            dt = self.dt_baseline   #wie viel von der baseline mitgegeben wird
            signal = np.array(self.signal[ev_start - dt: ev_end + dt])
            try:

                event_to_add = EventReload(signal, local_baseline=local_baseline, local_std=local_std,
                                           ev_indices=(ev_start, ev_end), fit_method=self.fit_method, **self.fit_kwargs)

                if Events.show:
                    ######## das zum plotten vorläufig ##########################
                    signal = event_to_add.corrected_signal
                    time = np.linspace(0, len(signal) / self.samplerate, len(signal))
                    print(event_to_add.lvls_info)
                    plt.plot(time, signal)
                    plt.xlabel('Time (s)')
                    plt.ylabel('Current (nA)')
                    for lvl_info in event_to_add.lvls_info:
                        plt.axhline(lvl_info[0], color="red")
                    plt.axvline(time[event_to_add.ev_start], color="green")
                    plt.axvline(time[event_to_add.ev_end], color="green")
                    plt.title('Event plot')
                    plt.legend()
                    plt.grid(True)
                    plt.show()
                    ########################################################

                #print(event_to_add.__dict__)
            except IndexError:  #wenn dies passiert, hat der pelt algo keine richtiges event detektiert
                continue
            except ValueError:          #passiert wenn 2 kurze lvls gelöscht werden -> keine entries mehr in max() bei height berechnung
                continue
            except NotFittable:         #mache dass alle exceptions hier rein gehen!
                continue
            except BadSegmentationParameters:   #das wenn segmentation nicht möglich (randbedingung)
                continue
            else:
                if event_to_add.ev_end - event_to_add.ev_start < 10:
                    continue
            events.append(event_to_add)

        return events

    def plot_event(self, n_ev):
        event_to_plot = self.events[n_ev]
        signal = event_to_plot.corrected_signal
        time = np.linspace(0, len(signal)/self.samplerate, len(signal))

        print(event_to_plot.lvls_info)

        plt.plot(time, signal)
        plt.xlabel('Time (s)')
        plt.ylabel('Current (nA)')
        for lvl_info in event_to_plot.lvls_info:
            plt.axhline(lvl_info[0], color="red")
        plt.axvline(time[event_to_plot.ev_start], color="green")
        plt.axvline(time[event_to_plot.ev_end], color="green")
        plt.title('Event plot')
        plt.legend()
        plt.grid(True)
        plt.show()


    #hier muss ich events bekommen. Dann die Event class callen und in dieser das cusum o.ä nutzen
    #denke wenn du event object kreierst, gebe für die events auch noch teile der baseline mit. also nicht nur das event


class EventReload:
    samplerate: float               #in SI
    samp_dwell_thresh: float      #in SI
    fit_time: float
    def __init__(self, signal, local_baseline, local_std, ev_indices, fit_method="cusum", **fit_kwargs):
        self.signal = signal
        self.local_std = local_std
        self.local_baseline = local_baseline
        self.ev_indices = ev_indices   #kann hier mitgeben wenn ich dwell von rough detec nehmen will
        self.fit_method = fit_method
        if EventReload.fit_time:
            self.check_fittable()      #mache hier besserer test ob man fitten kann
        if self.fit_method == "cusum":
            self.ct = self.cusum_lvl_fit(**fit_kwargs)      #hier einfach neuen algo mit neuen change times (achte drauf format wie bei cusum detec zu machen)
        elif self.fit_method == "pelt":
            self.ct = self.pelt(**fit_kwargs)
        elif self.fit_method == "dynamic":
            self.ct = self.dyn(**fit_kwargs)
        elif self.fit_method == "c_dynamic":
            self.ct = self.c_dyn(**fit_kwargs)
        elif self.fit_method == "c_pelt":
            self.ct = self.c_pelt(**fit_kwargs)
        self.set_feat()
        self.clean_short_lvls()  # das um zu schnelle lvlv zu löschen
        self.nr_lvls = len(self.lvls_info)
        if self.nr_lvls == 1:
            self.height = -1 * self.lvls_info[0][0] #(die -1 weil die levels immer relativ zur baseline eingetragen sind (d.h < 0)
        else:
            self.height = self.get_height()

    def check_fittable(self):
        if 3 < (self.ev_indices[0] - self.ev_indices[1]) <= (EventReload.fit_time * EventReload.samplerate):
            raise NotFittable("event not fittable")

    def cusum_lvl_fit(self, **fit_kwargs):
        hbook = fit_kwargs.get("hbook")
        delta = fit_kwargs.get("delta")
        sigma = fit_kwargs.get("sigma")
        if sigma:
            h = hbook * delta / sigma
        else:
            h = hbook * delta / self.local_std

        const_sig, det_t, ct = CUSUM(self.signal, delta, h)
        if ct[-1] - ct[0] < 10:                         #DAS ERSTMAL NUR VORLÄUFIG
            raise NotFittable("event not fittable")

        ####### mache hier vielleicht not impuls detection etc rein. Doch erst nach dem groben!
        return ct

    #hier andere algorithmen

    def pelt(self, **fit_kwargs):
        pen = fit_kwargs.get("pen")
        model = fit_kwargs.get("model", "l2")
        min_size = fit_kwargs.get("min_size_lvl", 3)

        ev = self.signal

        if pen == "BIC":
            pen = np.var(ev) * np.log(len(ev)) if model == "l2" else np.log(len(ev)) #BIC angepasst an l2 oder an normal (für den else fall wurde model = "normal" gewählt
        elif pen == "AIC":
            pen = np.var(ev)                                                         #AIC

        algo = rpt.Pelt(model=model, min_size=min_size, jump=1).fit(ev)
        ct = algo.predict(pen=pen)

        if Events.show:
            print(ct)
            rpt.display(ev, ct, figsize=(10, 6))
            plt.show()

        if ct[-2] - ct[0] < 10:                         #DAS ERSTMAL NUR VORLÄUFIG
            raise NotFittable("event not fittable")

        return ct[:-1]

    def c_pelt(self, **fit_kwargs):
        pen = fit_kwargs.get("pen")
        model = fit_kwargs.get("model", "l2")
        min_size = fit_kwargs.get("min_size_lvl", 3)

        ev = self.signal

        if pen == "BIC":    #das c_pelt funktioniert ja nur für cost l2
            pen = np.var(ev) * np.log(len(ev))
        elif pen == "AIC":
            pen = np.var(ev)

        #pen = np.var(ev) * np.log(len(ev)) if pen is None else pen

        if model == "l2":
            kernel = "linear"   #linear kernel ist entspricht l2 cost function (also gauß mean shit ohne varianz shift)
        else:
            raise NotImplemented("chosen model not implemented for KernelCPD algorithm yet.")
        #algo = rpt.Pelt(model=model, min_size=min_size, jump=1).fit(ev)
        algo = rpt.KernelCPD(kernel=kernel, min_size=min_size, jump=1).fit(ev)
        ct = algo.predict(pen=pen)

        if Events.show:
            print(ct)
            rpt.display(ev, ct, figsize=(10, 6))
            plt.show()

        if ct[-2] - ct[0] < 10:  # DAS ERSTMAL NUR VORLÄUFIG
            raise NotFittable("event not fittable")

        return ct[:-1]


    def dyn(self, **fit_kwargs):
        nr_ct = fit_kwargs.get("nr_ct")
        model = fit_kwargs.get("model", "l2")
        min_size = fit_kwargs.get("min_size_lvl", 3)

        ev = self.signal

        algo = rpt.Dynp(model=model, min_size=min_size, jump=1).fit(ev)        #hier vielleicht ändern, sd nur das ich nur das innere des event nehme hierfür
        bkps = algo.predict(n_bkps=nr_ct)

        if Events.show:
            print(bkps)
            rpt.show.display(ev, bkps, figsize=(10, 6))
            plt.title('Change Point Detection: Dynamic Programming Search Method')
            plt.show()

        if bkps[-2] - bkps[0] < 10:                         #DAS ERSTMAL NUR VORLÄUFIG
            raise NotFittable("event not fittable")

        return bkps[:-1]


    def c_dyn(self, **fit_kwargs):
        nr_ct = fit_kwargs.get("nr_ct")
        model = fit_kwargs.get("model", "l2")
        min_size = fit_kwargs.get("min_size_lvl", 3)

        ev = self.signal

        if model == "l2":
            kernel = "linear"
        else:
            raise NotImplemented("chosen model not implemented for KernelCPD algorithm yet.")
        algo = rpt.KernelCPD(kernel=kernel, min_size=min_size, jump=1).fit(ev)
        ct = algo.predict(n_bkps=nr_ct)

        if Events.show:
            print(ct)
            rpt.display(ev, ct, figsize=(10, 6))
            plt.show()

        if ct[-2] - ct[0] < 10:  # DAS ERSTMAL NUR VORLÄUFIG
            raise NotFittable("event not fittable")

        return ct[:-1]


    # def dynamic_pelt(self, **fit_kwargs):        #das scheint richtig gut!! Mache noch impult fit rein!!
    #     pen = fit_kwargs.get("pen")
    #     model = fit_kwargs.get("model", "l2")
    #     min_size = fit_kwargs.get("min_size_lvl", 3)
    #
    #     ev = self.signal
    #     # model = "rbf"            # mögl: rbf, normal, linear, ar, l1, l2
    #     algo = rpt.Pelt(model=model, min_size=min_size).fit(ev)
    #     ct = algo.predict(pen=pen)  # das ist penalty value, wahrscheinlich hängt es mit der threshhold zusammen. Hier war pen vorher 20
    #
    #     if Events.show:
    #         print(ct)
    #         rpt.display(ev, ct, figsize=(10, 6))
    #         plt.show()
    #
    #     nr_ct = len(ct) - 1
    #     # model = "rbf"
    #     algo = rpt.Dynp(model=model, min_size=3, jump=1).fit(ev)        #hier vielleicht ändern, sd nur das ich nur das innere des event nehme hierfür
    #     bkps = algo.predict(n_bkps=nr_ct)
    #
    #     if Events.show:
    #         print(bkps)
    #         rpt.show.display(ev, bkps, figsize=(10, 6))
    #         plt.title('Change Point Detection: Dynamic Programming Search Method')
    #         plt.show()
    #
    #     if bkps[-2] - bkps[0] < 10:                         #DAS ERSTMAL NUR VORLÄUFIG
    #         raise NotFittable("event not fittable")
    #
    #     return bkps[:-1]


    def set_feat(self):
        self.baseline = np.mean(self.signal[0:self.ct[0]])
        self.ev_start = self.ct[0]
        self.ev_end = self.ct[-1]
        self.dwell = (self.ev_end - self.ev_start)/EventReload.samplerate
        self.corrected_signal = list(map(lambda x: x-self.baseline, self.signal))
        self.mean = np.abs(np.mean(self.corrected_signal[self.ev_start:self.ev_end+1]))
        self.std = np.std(self.corrected_signal[self.ev_start:self.ev_end+1])

        lvls_info = []
        for i in range(1, len(self.ct)):
            lvl_curr = np.mean(self.signal[self.ct[i - 1]:self.ct[i]]) - self.baseline        #auch experimentell wegen derm +1 von oben
            # lvl_curr = np.mean(const_sig[ct[i - 1]:ct[i]]) - self.baseline
            lvl_dwell_sample = self.ct[i] - self.ct[i-1]
            if lvl_dwell_sample <= EventReload.samp_dwell_thresh:   #Für PoreJ mache hier 2 statt 7                     #HIER FÜR CLEAN SHORT LEVELS
                continue
            else:
                lvl_dwell = lvl_dwell_sample/EventReload.samplerate
            lvls_info.append((lvl_curr, lvl_dwell))

        self.lvls_info = lvls_info


    def clean_short_lvls(self):
        lvls_info = self.lvls_info
        if len(lvls_info) == 1:
            return
        else:
            for i in range(len(lvls_info)-1, -1, -1):
                if lvls_info[i][1] < EventReload.samp_dwell_thresh/EventReload.samplerate:                          #HIER LÖSCHE AUF ZU KURZ BASIS IN DWELL THRESH BASIS
                    del lvls_info[i]
                # if lvls_info[i][0] > -0.2:     #hier um lvl über baseline zu löschen, der algo ist noch nicht ganz ausgereift
                #     del lvls_info[i]
            self.lvls_info = lvls_info

    def get_height(self):
        levels_cleaned = [lvl for lvl, _ in self.lvls_info]
        height = max(levels_cleaned) - min(levels_cleaned)
        return height





def get_events(signal, samplerate, samp_dwell_thresh, fit_method="pelt", show=False, rough_detec_params={}, **fit_kwargs):
    # b, a = sig.butter(order, freq, btype=btype, output="ba")   #bei freq mache gewünschte freq scaled auf nyquist freq, also 2*freq_gewün/samplerate
    # b, a = sig.bessel(order, freq, btype=btype, output="ba")
    # signal = sig.filtfilt(b, a, signal.reshape(1, -1))


    #hier könnte signale in subsignals machen, um memory zu sparen.
    #denn sonst kackt die std berechnung ab. Die einzelnen dev dinge klappen aber!!

    rough_detec_params["samplerate"] = samplerate
    rd_algo = rough_detec_params.get("rd_algo")
    dt_baseline = rough_detec_params.get("dt_baseline", 50)
    # rough_detec_algorithm = RecursiveLowPassFast if rd_algo == "lowpass" else find_rough_event_loc
    rough_detec_algorithm = find_rough_event_loc if rd_algo == "exact" else RecursiveLowPassFast

    pre_rough_time = tm.time()
    event_info = rough_detec_algorithm(signal, **rough_detec_params)
    post_rough_time = tm.time()
    print("elapsed time: ", post_rough_time - pre_rough_time)
    print("nr events (rough detec): ", len(event_info))

    Events.show = show      #hier ob geplotted werden soll

    fitted_events = Events(event_info, samplerate=samplerate, samp_dwell_thresh=samp_dwell_thresh, dt_baseline=dt_baseline,   #vorher dt_baseline=50
                           signal=signal, fit_method=fit_method, **fit_kwargs)

    ###################### das für running time berechnung. Mache einfach raus falls ich nicht mehr will ################################
    # post_fine_time = tm.time()
    # nr_points = sum(map(lambda x: 2*dt_baseline + (int(x[1]) - int(x[0])), event_info))
    # with open(r"running_times/running_times_chip_2.json", 'r+') as file:  # für running time calculation
    #     data = json.load(file)
    #     if data.get(fit_method):
    #         data[fit_method].append((nr_points, post_fine_time-pre_rough_time))     #hier entweder pre_rough_time oder post_rough_time
    #     else:
    #         data[fit_method] = [(nr_points, post_fine_time-pre_rough_time)]
    #     file.seek(0)  # cursor geht an den anfang
    #     json.dump(data, file)
    #     file.truncate()  # der alte content wird gelöscht
    ########################### bis hier für running time berechnung ###########################
    # exit()                                      #DAS WEG MACHEN WENN MIT RUNNING TIME FERTIG

    features_df = pd.DataFrame(columns=["dwell_time", "mean", "height", "num_lvls", "std"])
    lvl_feat_df = pd.DataFrame(columns=["lvl_dwells", "lvl_drops"])

    for event_to_add in fitted_events:
        if not event_to_add.height: continue                                    # hier zum überspringen von events, mit zu kurzer dwell, gucke oben in class
        if event_to_add.dwell > 0.1 or event_to_add.height < 0: continue   #hier um events mit zu langen dwells zu streichen und um verkackte heights zu streichen
        new_row = {"dwell_time": event_to_add.dwell, "mean": event_to_add.mean,
                   "height": event_to_add.height, "num_lvls": event_to_add.nr_lvls, "std": event_to_add.std}
        features_df.loc[len(features_df)] = new_row

        for i in range(0, len(event_to_add.lvls_info)):
            new_lvls_row = {"lvl_dwells": event_to_add.lvls_info[i][1], "lvl_drops": event_to_add.lvls_info[i][0]}           #DAS NEU
            lvl_feat_df.loc[len(lvl_feat_df)] = new_lvls_row

    return features_df, lvl_feat_df










