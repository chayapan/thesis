FROM chayapan/jupyterlab
# This docker image is for running on Windows and Ubuntu.
# The base imgage chayapan/jupyterlab defined in tmm-1/spark-cluster

RUN pip3 install seaborn
RUN pip3 install sqlalchemy
RUN pip3 install psycopg2-binary
RUN pip3 install plotly
RUN pip3 install requests>=2.23.0
RUN pip3 install numpy
RUN pip3 install pandas
RUN pip3 install pandas-datareader
RUN pip3 install scikit-learn==0.24
RUN pip3 install scipy>=1.4.1
RUN pip3 install statsmodels==0.12.1
RUN pip3 install lxml==4.5.0
RUN pip3 install sqlparse==0.4.1
RUN pip3 install sympy==1.5.1
RUN pip3 install tabulate==0.8.7
RUN pip3 install tblib==1.7.0
RUN pip3 install tensorboard>=2.3.0
RUN pip3 install tensorboard-plugin-wit>=1.7.0
RUN pip3 install grpcio>=1.32.0
RUN pip3 install six>=1.15.0
RUN pip3 install wheel>=0.35
RUN pip3 install tensorflow==2.0.0b1
RUN pip3 install xlrd
RUN pip3 install dtw-python fastdtw


# -- Runtime
# chayapan/jupyterlab defined in tmm-1/spark-cluster

# EXPOSE 8888
# WORKDIR ${SHARED_WORKSPACE}
# CMD jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token=
