Introduction:
In this project, the improvements of energy consumption is focussed with the ASHRAE -
Great Energy Predictor III dataset obtained from kaggle platform. The lightGBM model (a
Microsoft open source library) is constructed on this dataset to predict the meter_reading. At first
to get the important predictor variables, the exploratory data analysis part is executed. After
gathering enough information, the model is constructed and the dataset is given to train the
model. This model gets trained and the dataset is given for prediction. The resultant prediction
values is then exported in the required format.
Handling missing values:
Firstly, the dataset is fed into the notebook using pandas library. The dataset is then
checked for its values using ‘.info()’ methods and ‘.isna()’ methods. The site_id and timestamp
do not have any missing values. But the variables such as ‘ precip_depth_1_hr’ and
‘cloud_coverage’ have missing values in them. Since ‘floor_count’ has a larger number of missing
values, this column is dropped from the dataset. The missing values are estimated with the mean
values.
Exploratory data analysis:
Using the pandas library again, the dataset is explored using ‘.describe()’ method and
the values of each variable is analysed. Here, one data preparation task is done where the
variable ‘meter’ is slightly modified with named values such as ‘Electricity’, ‘Chilled Water’,
‘Steam’, ‘Hot Water’. This modified column is used to get the number of unique building types
with the distribution of meter id.

Hot water meter reading is high during the winter months and reduces during the summer months.
Finding Important Variables:
The important columns are drawn using the correlation matrix and the threshold value
0.9. If the threshold value is less than the correlation value of the variable and the meter reading
then the corresponding variable is dropped. Also, the important variables are proofread using
the regression feature importance variables (LightGBM model).
Feeding the dataset to LightGBM model:
The dataset is split using the train_test_split method with the test size as 0.25. Later, the
split train and test datasets are given to the ‘lgb.train()’ method and the dataset is trained. The
root mean square error value gets reduced at the 3000th level.
Prediction:
The ‘prediction.extend()’ method is used to predict the meter_reading for the test dataset
with the step count as 10000. This process takes a CPU time of 5hours 7mins and 33s, wall
time as 44min and 19s. Also, the required format is printed in the required format using the
‘to_csv()’ method which took a CPU time of 2min 5s and it’s the same for wall time as well.
Instructions to run:
1. Install python version 3 on your Mac/Windows operating system.
2. Install the required packages using ‘pip install’.
3. The required packages are listed below
['builtins','builtins', 'os', 'numpy', 'pandas', 'matplotlib.pyplot', 'seaborn', 'datetime',
'gc','lightgbm', 'pip', 'types'].
4. Run the python file from command line/Jupyter Notebook/Pycharm IDE
Output:
3). Sample Output
