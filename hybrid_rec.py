# Method Description: Initial hybrid recommendation has RMSE of 0.9899172749284479. To improve upon the accuracy of
# the hybrid recommendation system, I first experimented with GridSearchCV to tune the hyperparameters of the
# XGBRegressor. This didn't improve the accuracy by much, so I looked through the feature extraction code I have and
# found that I was converting a lot of the numbers into integers. I experimented with changing datatype to float,
# and it dropped the RMSE to 0.9838187684904954. As this is still unstatisfactory, I experimented with adding more
# features from business.json, tip.json, photo.json, and user.json. Main focus was to add in additional features to
# help XGBRegressor predict the ratings. Finally, I have also chosen to include predictions from item-based CF as a
# feature instead of using a weighted average approach. To round it out, I used GridSearchCV to finalize the
# hyperparameters of the XGBRegressor model locally. I was able to achieve a RMSE of 0.9785272832228583 using this
# method.

# Error Distribution:
# >=0 and <1: 102200
# >=1 and <2: 32910
# >=2 and <3: 6138
# >=3 and <4: 794
# >=4: 2

# RMSE:
# 0.9785272832228583

# Execution Time:
# 261.8420760631561

from pyspark import SparkContext
import sys
import time
import json
from math import nan
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV

start = time.time()

folder_path = sys.argv[1]
test_file = sys.argv[2]
output_file = sys.argv[3]

sc = SparkContext('local[*]', 'hybrid_rec')
sc.setLogLevel("ERROR")


def getWeight(pair):
    """
    For item-based recommender.
    Calculate the weight and return the rating and the weight in a list of tuples, ie [rating, weight].
    Depending on number of common users between the target business and other business, there will be 4 ways of handling
    weight calculations:
    1) When number of common users is at least 5, co-rated average is used to calculate Pearson's correlation.
    2) When number of common users is between 2 and 5, all-rated average is used to calculate Pearson's correlation.
    3) When number of common user is at 2, use Min Max difference normalization to calculate weight, with both users
    giving contributing equally.
    4) When there is one or no common user, single Min Max difference normalization to calculate weight.
    """
    # Business_id and user_id from test set
    target_bus, target_user = pair

    rating_weight = []

    for other_bus in group_by_user[target_user]:
        # Get the list of users for other business
        sorted_bus_pair = tuple(sorted((other_bus, target_bus)))
        if sorted_bus_pair in weight_dict.keys():
            weight = weight_dict[sorted_bus_pair]
        else:
            # Generate a list of common users between the two businesses
            common_users = group_by_bus[other_bus].intersection(group_by_bus[target_bus])
            len_common_user = len(common_users)

            # Define the maximum possible difference since rating is from 0 to 5 for MinMax normalization
            max_diff = 5.0

            if len_common_user > 2:
                # Calculate Pearson with all-rated average
                target_bus_ratings = []
                other_bus_ratings = []
                for user_id in common_users:
                    target_bus_ratings.append(float(rating_dict[target_bus][user_id]))
                    other_bus_ratings.append(float(rating_dict[other_bus][user_id]))
                avg_target_bus = sum(target_bus_ratings) / len(target_bus_ratings)
                avg_other_bus = sum(other_bus_ratings) / len(other_bus_ratings)
                target_bus_diff = [r - avg_target_bus for r in target_bus_ratings]
                other_bus_diff = [r - avg_other_bus for r in other_bus_ratings]

                numerator = (sum([x * y for x, y in zip(target_bus_diff, other_bus_diff)]))
                denominator_1 = sum([x ** 2 for x in target_bus_diff]) ** 0.5
                denominator_2 = sum([y ** 2 for y in other_bus_diff]) ** 0.5

                if denominator_1 == 0 or denominator_2 == 0:
                    weight = 0
                else:
                    weight = numerator / (denominator_1 * denominator_2)

            elif len_common_user == 2:
                # Special case for 2 common users, use MinMax normalization
                common_users = list(common_users)
                delta_1 = max_diff - abs(float(rating_dict[target_bus][common_users[0]]) -
                                         float(rating_dict[other_bus][common_users[0]]))
                delta_2 = max_diff - abs(float(rating_dict[target_bus][common_users[1]]) -
                                         float(rating_dict[other_bus][common_users[1]]))
                normalized_d1 = delta_1 / max_diff
                normalized_d2 = delta_2 / max_diff
                weight = (normalized_d1 + normalized_d2) / 2

            else:
                # Use MinMax normalization for one or less common user
                delta = abs(bus_avg_dict[target_bus] - bus_avg_dict[other_bus])
                weight = 1 - delta / max_diff
            weight_dict[sorted_bus_pair] = weight
        # Generate a dictionary with key as the user rating and weight as the value
        rating_weight.append((float(rating_dict[other_bus][target_user]), weight))
    return rating_weight


def makePred(pair):
    """
    For item-based recommender.
    Calculate the predicted rating for the pair of business_id and user_id passed in.
    Get the rating and weight from getWeight() and take top 14 weights to make predictions.
    Predictions are calculated based on the equation given in class.
    """
    rating_weight = getWeight(pair)
    sorted_weights = sorted(rating_weight, key=lambda item: item[1], reverse=True)
    # Test locally to determine ideal size of neighborhood
    top_weights = sorted_weights[:15]

    numerator, denominator, prediction = 0, 0, 0
    for rating, weight in top_weights:
        numerator += rating * weight
        denominator += abs(weight)

    if denominator != 0:
        prediction = numerator / denominator
    else:
        prediction = 3.0

    return prediction


def checkColdStart(pair):
    """
    For item-based recommender.
    Start of the recommendation system. Check if target passes in are cold starts.
    There are 3 scenarios handled in this function:
    1) If the user_id in test data doesn't exist in train data, then use global user average to predict rating.
    2) if bus_id doesn't exist in train data but user_id does, then use the individual user's average rating for
    existing businesses to predict rating.
    3) If both exist, initialize the item-based CF recommendation system.
    """
    bus_id, user_id = pair

    if user_id not in group_by_user.keys():
        # When use is not in train data, use global average of all rating in the system
        cold_start = 3.0
        return user_id, bus_id, cold_start
    if bus_id not in group_by_bus.keys():
        # When business is not in train but user is, use user average
        return user_id, bus_id, user_avg_dict[user_id]

    # If user_id and bus_id exist in train data, start item-based CF
    prediction = makePred(pair)
    return user_id, bus_id, prediction


def getXY(row):
    """
    For Model Based recommender - XGBRegressor.
    Create the X and y for XGBoost Regressor with training data and create only X for testing data.
    """

    # Use math.nan here because numpy.NaN is not allowed in original HW3
    bus_stars, bus_review_cnt, lat, long, is_open, attribute = nan, nan, nan, nan, nan, nan
    kid, seating, delivery, bike, card, tv, noise, groups, price = nan, nan, nan, nan, nan, nan, nan, nan, nan
    reserve, table, takeout, wifi = nan, nan, nan, nan
    avg_useful, avg_funny, avg_cool = nan, nan, nan
    checkin, bus_photo, tip_no = nan, nan, nan
    user_review_cnt, user_fans, user_avg_stars, user_useful, user_funny, user_cool = nan, nan, nan, nan, nan, nan
    comp_hot, comp_more, comp_prof, comp_cute, comp_list, comp_note = nan, nan, nan, nan, nan, nan
    comp_plain, comp_cool, comp_funny, comp_writer, comp_photos, yelping_years, year = nan, nan, nan, nan, nan, nan, nan
    item_based_rating, knn_rating = nan, nan

    if len(row) == 3:
        # For train data
        user_id, bus_id, rating = row
        y_data = float(rating)
        # Include item-based predictions as features
        if (user_id, bus_id) in item_result_train.keys():
            item_based_rating = item_result_train[(user_id, bus_id)]
    else:
        # For test data
        user_id, bus_id = row
        y_data = None
        # Include item-based predictions as features
        if (user_id, bus_id) in item_result_test.keys():
            item_based_rating = item_result_test[(user_id, bus_id)]

    # Prepare all possible features and create X for XGBRegressor
    if bus_id in bus_data.keys():
        bus_stars, bus_review_cnt, lat, long, is_open, attribute = bus_data[bus_id]
        if attribute is not None:
            # Extract interesting features under attributes tag
            if attribute.get('GoodForKids', False) == 'True':
                kid = 1
            else:
                kid = 0
            if attribute.get('OutdoorSeating', False) == 'True':
                seating = 1
            else:
                seating = 0
            if attribute.get('RestaurantsDelivery', False) == 'True':
                delivery = 1
            else:
                delivery = 0
            if attribute.get('BikeParking', False) == 'True':
                bike = 1
            else:
                bike = 0
            if attribute.get('BusinessAcceptsCreditCards', False) == 'True':
                card = 1
            else:
                card = 0
            if attribute.get('HasTV', False) == 'True':
                tv = 1
            else:
                tv = 0
            if attribute.get('RestaurantsGoodForGroups', False) == 'True':
                groups = 1
            else:
                groups = 0
            price = float(attribute.get('RestaurantsPriceRange2', False))

            if attribute.get('NoiseLevel', False) == 'quiet':
                noise = 0
            elif attribute.get('NoiseLevel', False) == 'average':
                noise = 1
            elif attribute.get('NoiseLevel', False) == 'loud':
                noise = 2
            elif attribute.get('NoiseLevel', False) == 'very_loud':
                noise = 3
            if attribute.get('RestaurantsReservations', False) == 'True':
                reserve = 1
            else:
                reserve = 0
            if attribute.get('RestaurantsTableService', False) == 'True':
                table = 1
            else:
                table = 0
            if attribute.get('RestaurantsTakeOut', False) == 'True':
                takeout = 1
            else:
                takeout = 0
            if attribute.get('WiFi', False) == 'free':
                wifi = 1
            else:
                wifi = 0

    # Extract remaining business related features
    if bus_id in review.keys():
        avg_useful, avg_funny, avg_cool = review[bus_id]
    if bus_id in checkins.keys():
        checkin = checkins[bus_id]
    if bus_id in photo.keys():
        bus_photo = photo[bus_id]

    # Extract user related features
    if user_id in user_data.keys():
        user_review_cnt, user_fans, user_avg_stars, user_useful, user_funny, user_cool, comp_hot, comp_more, comp_prof, \
        comp_cute, comp_list, comp_note, comp_plain, comp_cool, comp_funny, comp_writer, \
        comp_photos, yelping_years = user_data[user_id]
        year = 2024 - int(yelping_years[:4])

    # Extract remaining interesting feature
    if (user_id, bus_id) in tips.keys():
        tip_no = tips[(user_id, bus_id)]

    # Totally using 42 features
    X_data = [item_based_rating, bus_stars, bus_review_cnt, lat, long, is_open, kid, seating, delivery, bike, card, tv,
              noise, groups, price, reserve, table, takeout, wifi, avg_funny, avg_cool, checkin, bus_photo,
              user_review_cnt, user_fans, user_avg_stars, user_useful, user_funny, user_cool, comp_hot, comp_more,
              comp_prof, comp_cute, comp_list, comp_note, comp_plain, comp_cool, comp_funny, comp_writer, comp_photos,
              year, tip_no]

    return X_data, y_data


# Read in train data and test data
train_file = folder_path + '/yelp_train.csv'
raw_train = sc.textFile(train_file)
train_head = raw_train.first()
train_data = raw_train.filter(lambda row: row != train_head).map(lambda row: row.split(','))
raw_test = sc.textFile(test_file)
test_head = raw_test.first()
test_data = raw_test.filter(lambda row: row != test_head).map(lambda row: row.split(','))

# Define dictionaries for Item-based CF
group_by_user = train_data.map(lambda row: (row[0], row[1])).groupByKey().mapValues(set).collectAsMap()
group_by_bus = train_data.map(lambda row: (row[1], row[0])).groupByKey().mapValues(set).collectAsMap()
user_avg_dict = train_data.map(lambda row: (row[0], float(row[2]))) \
    .groupByKey() \
    .mapValues(list) \
    .map(lambda row: (row[0], sum(row[1]) / len(row[1]))) \
    .collectAsMap()
bus_avg_dict = train_data.map(lambda row: (row[1], float(row[2]))) \
    .groupByKey() \
    .mapValues(list) \
    .map(lambda row: (row[0], sum(row[1]) / len(row[1]))) \
    .collectAsMap()
rating_dict = train_data.map(lambda row: (row[1], (row[0], row[2]))) \
    .groupByKey() \
    .mapValues(dict) \
    .collectAsMap()

# Start item-based CF on train data, for features in model-based.
weight_dict = {}
item_result_train = train_data.map(lambda row: (row[0], row[1])) \
    .map(checkColdStart) \
    .map(lambda row: ((row[0], row[1]), row[2])) \
    .collectAsMap()
# Start item-based CF on test data
weight_dict = {}
item_result_test = test_data.map(lambda row: (row[0], row[1])) \
    .map(checkColdStart) \
    .map(lambda row: ((row[0], row[1]), row[2])) \
    .collectAsMap()

# Define features for Model Based Recommender
bus_json = folder_path + '/business.json'
bus_data = sc.textFile(bus_json).map(json.loads) \
    .map(lambda row: (row['business_id'], (float(row['stars']),
                                           float(row['review_count']),
                                           row['latitude'],
                                           row['longitude'],
                                           row['is_open'],
                                           row.get('attribute', {})))) \
    .collectAsMap()
review_file = folder_path + '/review_train.json'
review = sc.textFile(review_file).map(json.loads) \
    .map(lambda row: (row['business_id'], (float(row['useful']), float(row['funny']),
                                           float(row['cool']), 1))) \
    .reduceByKey(lambda x, y: (x[0] + y[0], x[1] + y[1], x[2] + y[2], x[3] + y[3])) \
    .mapValues(lambda v: (v[0] / v[3], v[1] / v[3], v[2] / v[3])) \
    .collectAsMap()
user_file = folder_path + '/user.json'
user_data = sc.textFile(user_file).map(json.loads) \
    .map(lambda row: (row['user_id'], (float(row['review_count']), float(row['fans']),
                                       float(row['average_stars']), float(row['useful']),
                                       float(row['funny']), float(row['cool']),
                                       float(row['compliment_hot']),
                                       float(row['compliment_more']),
                                       float(row['compliment_profile']),
                                       float(row['compliment_cute']),
                                       float(row['compliment_list']),
                                       float(row['compliment_note']),
                                       float(row['compliment_plain']),
                                       float(row['compliment_cool']),
                                       float(row['compliment_funny']),
                                       float(row['compliment_writer']),
                                       float(row['compliment_photos']),
                                       row['yelping_since']))) \
    .collectAsMap()
checkin_json = folder_path + '/checkin.json'
checkins = sc.textFile(checkin_json).map(json.loads) \
    .map(lambda row: (row['business_id'], row['time'])) \
    .flatMapValues(lambda x: x.values()) \
    .map(lambda x: (x[0], int(x[1]))) \
    .reduceByKey(lambda x, y: x + y) \
    .collectAsMap()
photos_json = folder_path + '/photo.json'
photo = sc.textFile(photos_json).map(json.loads) \
    .map(lambda row: (row['business_id'], 1)) \
    .reduceByKey(lambda x, y: x + y) \
    .collectAsMap()
tip_json = folder_path + '/tip.json'
tips = sc.textFile(tip_json).map(json.loads) \
    .map(lambda row: ((row['user_id'], row['business_id']), 1)) \
    .reduceByKey(lambda x, y: x + y) \
    .collectAsMap()

# Generate X and y for training data for XGBRegressor
model_train = train_data.map(getXY)
X_train = [row[0] for row in model_train.collect()]
y_train = [row[1] for row in model_train.collect()]
# Generate X for test data
test_features = test_data.map(getXY)
X_test = [row[0] for row in test_features.collect()]

# # GridSearch locally for the best combination of hyperparameters
# grid = {
#     'learning_rate': [0.17, 0.18, 0.19],
#     'max_depth': [5, 6, 7],
#     'reg_alpha': [0.4, 0.6, 0.8],
#     'reg_lambda': [0.4, 0.6, 0.8]
# }
# xgb = XGBRegressor()
# search = GridSearchCV(xgb, grid, scoring='neg_mean_squared_error')
# search.fit(X_train, y_train)
# print(search.best_params_)
# y_pred = search.predict(X_test)

# Start XGBRegressor based on parameters determined by GridSearch
param = {
    'learning_rate': 0.18,
    'max_depth': 6,
    'reg_alpha': 0.6,
    'reg_lambda': 0.6
}
xgb = XGBRegressor(**param)
xgb.fit(X_train, y_train)
y_pred = xgb.predict(X_test)

# Make sure every prediction is between 1 and 5
pred_range = np.clip(y_pred, 1, 5)

# Prepare for output
model_result = test_data.zipWithIndex().map(lambda x: (x[0][0], x[0][1], pred_range[x[1]])) \
    .map(lambda x: ",".join(map(str, x)))

# Output to file
csv_output = sc.parallelize(["user_id, business_id, prediction"]).union(model_result)
with open(output_file, 'w') as csv_file:
    csv_file.write("\n".join(csv_output.collect()))

print("Duration:", time.time() - start)
