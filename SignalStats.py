"""Provides tools for computing signal stats.
Gives tolls for deviding signal data into sections and calculate statistical measures for them.
It also provides tools for alerting and computing welding samples imperfections data,
based on given imperfection database.
"""

import numpy as np
import pandas as pd

class Window():
    """Calculate signal stats for window"""
    def __init__(self, sig):
        """
        Keyword argument:
        sig -- processed signal
        """
        self.sig = sig
        self.med = []    
        self.q25 = []    
        self.q75 = []    
        self.std = []    
        self.skew = []   
        self.kurt = []   
        self.time = []   #computed timesteps
        self.min = []    
        self.max = []    
        self.rms = []    
        self.var = []    
        self.mean = []   
        self.mode = []   

    def win_clc(self, sig):
        """Calculate statistical measures for each section
        
        Keyword argument:
        sig -- processed signal
        """
        for a in sig:
            self.med.append(a.median())
            self.q25.append(a.quantile(.25))
            self.q75.append(a.quantile(.75))
            self.std.append(a.std())
            self.skew.append(a.skew())
            self.kurt.append(a.kurtosis())
            self.min.append(a.min())
            self.max.append(a.max())
            self.rms.append(np.sqrt(np.mean(np.square(a))))
            self.var.append(a.var())
            self.mean.append(a.mean())
            self.mode.append(a.mode())

    def time_col(self, first, l, t_step):
        """Generate timesteps according to ammount of sections
        
        Keyword arguments:
        first -- starting time value
        l -- ammount of sections
        t_step -- length of the timestep
        """
        for a in range(l):
            self.time.append(first + (a + 1) * t_step)

    '''metoda dzieląca sygnał na segmenty'''
    def win_split(self, l):
        """Divide signal into segments and return as list
        
        Keyword argument:
        l -- sections length
        """
        i = len(self.sig) #samples quantity
        it = int(i / l) #sections quantity
        return np.array_split(self.sig, it)
    
    def get_stat(self, dist1, dist2):
        """Return all calculated statistical measures for given range
        
        Keyword arguments:
        dist1 -- starting time value
        dist2 -- ending time value
        """
        dist1 = dist1*2 - 1
        dist2 = dist2*2 - 1
        return [
            self.med[dist1:dist2], self.q25[dist1:dist2], self.q75[dist1:dist2], 
            self.std[dist1:dist2], self.kurt[dist1:dist2], self.min[dist1:dist2], 
            self.max[dist1:dist2], self.rms[dist1:dist2], self.var[dist1:dist2], 
            self.mean[dist1:dist2], self.mode[dist1:dist2]
            ]
    
class Main_sig():
    """Calculate signal stats for whole signal"""
    def __init__(self, sig):
        """
        Keyword argument:
        sig -- processed signal
        """
        self.sig = sig

    def stat_clc(self):
        """Calculate statistical measures"""
        self.qn75 = self.sig.quantile(.75)
        self.qn25 = self.sig.quantile(.25)
        self.stdev = self.sig.std()
        self.qn50 = self.sig.quantile(.50)
        self.rms = np.sqrt(np.mean(np.square(self.sig)))

class Imperfection():
    """Handle welding samples imperfections"""
    def __init__(self, db, mark, tab):
        """
        Keyword arguments:
        db -- imperfections database
        mark -- welding sample mark
        tab -- specific table in database
        """
        self.db = pd.read_sql_query(f"SELECT * FROM {tab} where mark='{mark}'", db)

    def maxDist(self):
        """Return ending position of last imperfection"""
        dist = self.db['distance'].tolist() + self.db['distance2'].tolist()
        return max(dist)
    
    def minDist(self):
        """Return starting position of first imperfection"""
        dist = self.db['distance'].tolist() + self.db['distance2'].tolist()
        return min(dist)
    
    def lenAdjust(self, max, l):
        """Check if ending postion is out of the signal range
        
        Keyword arguments:
        max -- signal range
        l -- given position
        """
        if max > l:
            return max - l
        else:
            return 0 
        
    def std_min_alert(self, std, qn50, min, step):
        """Return lowest standard deviation level exeeded by the signal value
        
        Keyword arguments:
        std -- standard deviation value
        qn50 -- median value
        min -- minimal signal value
        step -- range between standard deviation levels
        """
        i = 1
        for s in np.arange(std, 5*std, step):
            if min < qn50 - s and not min >= qn50 - (s - step): #check if value is in given range
                return i
            i = i+1
        if min >= qn50 - std:
            return 0
        else:
            return i  #if the value is lower then the minimal std level (5*std)

    def std_max_alert(self, std, qn50, max, step):
        """Return highest standard deviation level exeeded by the signal value
        
        Keyword arguments:
        std -- standard deviation value
        qn50 -- median value
        max -- maximal signal value
        step -- range between standard deviation levels
        """
        i = 1
        for s in np.arange(std, 5*std, step):
            if max > qn50 + s and not max <= qn50 + (s + step): #check if value is in given range
                return i
            i = i+1
        if max <= qn50 + std:
            return 0
        else:
            return i  #if the value is higher then the maximal std level (5*std)
        
    def kurt_alert(self, kurt, step):
        """Return highest level exeeded by the kurtosis value
        
        Keyword arguments:
        kurt -- standard deviation value
        step -- range between kurtosis levels
        """
        i = 1
        for k in np.arange(-1, 4, step):
            if kurt > k and not kurt >= (k + step): #check if value is in given range
                return i
            i = i+1
        if kurt <= -1:
            return 0
        else:
            return i  #if the value is higher then the maximal kurtosis level (kurtosis=4)
        
    def rms_max_alert(self, rms, max):
        """Check if the given signal value is higher then the RMS value
        
        Keyword arguments:
        rms -- RMS value
        max -- maximal signal value in processed window       
        """
        if max < rms:
            return 1
        else:
            return 0
        
    def rms_min_alert(self, rms, min):
        """Check if the given signal value is lower then the RMS value
        
        Keyword arguments:
        rms -- RMS value
        min -- minimal signal value in processed window       
        """
        if min > rms:
            return 1
        else:
            return 0
    
    def imp_iter(self, l):
        """Generate imperfections list for processed welding sample
        
        Keyword argument:
        l -- signal length
        """
        imp_list = []
        for a, b in self.db.iterrows():
            if not np.isnan(b['distance2']):
                imp_list.append( [b['distance'] * 2, b['distance2'] * 2, b['Imperfection_description'], b['Imperfection_id']] )
            else:
                #if distance value is not out of range
                if not self.lenAdjust(self.maxDist(), l):
                    imp_list.append( [b['distance'] * 2, None, b['Imperfection_description'], b['Imperfection_id']])
                else:
                    print('Out of range')
        return imp_list