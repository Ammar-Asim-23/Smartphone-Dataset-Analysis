# Smartphone-Dataset-Analysis

We have taken a Smartphones dataset which consists of 210 rows and 26 different columns . The columns includes Brand Name , model , price , rating , has_5g, has_nfc etc and
Row consists of different 210 phones with models and brands. We will perform analysis and prediction model.

# Data Analysis

We have analyzed that the following 17 attributes are important for the price prediction in this dataset:
1. resolution_height
2. processor_speed
3. screen_size
4. internal_memory
5. resolution_width
6. primary_camera_front
7. rating
8. ram_capacity
9. has_nfc
10. extended_memory_available
11. primary_camera_rear
12. os_ios
13. refresh_rate
14. has_5g
15. num_rear_cameras
16. num_cores
17. processor_brand_snapdragon

# Model Analysis

Until now we have trained several models and calculated their r2_scores on this dataset. Here we will list the models and their r2 scores:

1. LinearRegression (r2_score = 0.6574685071128396)
2. RandomForestRegressor (r2_score = 0.7479186726265188)
3. RandomizedSearchCV (r2_score = 0.8542147632441565)
4. GradientBoostingRegressor (r2_score = 0.6368128678048927)

Thus, we can infer that RandomizedSearchCV works best on this dataset.
