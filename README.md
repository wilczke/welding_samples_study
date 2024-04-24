# Welding samples study
This project contains tools for signal processing. They were originally used in welding signal analysis.

# Test stand
For the examination porpouse 50 samples were prepared, welded with diferent parameters. The samples were produced using KUKA KR 240 welding robot equiped with KRC2 control system.

The stand was equiped with current transformer LEM AHR 800 B10 and voltage transformer LEM LV 25-P. For the data acquisition propouse National Instruments NI USB-6008 card was used in combination with the Data Acquisition Toolbox for MATLAB.

# Usage example
in development...

```
import knn_test as kt
```

```
df = pd.read_csv('VTrecords4')
clas_l = [0, 1, 2, 3, 4, 5, 9, 11, 12, 13, 15, 17, 20, 21] 
features_l = ['range_V', 'iqr_V', 'std_V', 'skew_V', 'kurt_V', 'range_C', 'iqr_C', 'std_C', 'skew_C', 'kurt_C']
```

```
I2 = kt.KNN_test(df, clas_l, features_l)
I2.build_model(['Imperfection_id'], 0.3, 13)
```

```
a = I2.polt_conf_matrix(0)
```
![obraz](https://github.com/wilczke/welding_samples_study/assets/103566385/8a1a0b5e-b859-4f08-bcd4-3fa0406b030a)
