# ARIMA Error Distribution

Control Factors  
1. hyperparameters: (3,1,0), (4,1,0)  
2. history window: 100, 50, 30, 15, 10  
3. forecast horizon: 1, 3, 5,7, 14, 30

[Sketch notebook in Colab](https://colab.research.google.com/drive/1AHJjIbK3HwmuualpKO9vx5tpUhrYQSgK)

Use Virtualenv to run.

```
source ~/venv/bin/activate
python arima_errors.py


History: 30 days Forecast: 1 days
MAPE: 0.01173
Batches Tried: 1541
Took 65.21 sec.
History: 30 days Forecast: 3 days
MAPE: 0.02065
Batches Tried: 1539
Took 67.81 sec.
History: 30 days Forecast: 5 days
MAPE: 0.02700
Batches Tried: 1537
Took 67.76 sec.
...
```

```
# ANTONINOUS
virtualenv --python=/usr/local/bin/python3 ~/venv
pip install -r ../../requirements.txt
```

Or use chayapan/tmm-1 docker image and run under virtualenvironment. This is slow.


### output

This produces images to img/  
Save data to arima_errrors.csv
