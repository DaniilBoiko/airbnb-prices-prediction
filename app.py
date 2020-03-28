from flask import Flask, request
from geopy.geocoders import Nominatim
import pickle as pkl

app = Flask(__name__)
from flask import render_template
import pandas as pd
from random import randint
import numpy as np
from string import punctuation

from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer
import re, string, unicodedata
import inflect


@app.route('/index/')
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict/')
def predict():
    amenities = {'High chair',
                 'Smartlock', 'Fireplace guards', 'Heating',
                 'Cleaning before checkout', 'Smoke detector', 'Lockbox',
                 'Refrigerator', 'Hot water', 'Dryer', 'Self Check-In',
                 'Smoking allowed', 'Stove', 'Bed linens', 'Microwave', 'Crib',
                 'Washer / Dryer', 'Air conditioning', 'Fire extinguisher',
                 'Coffee maker', 'Ethernet connection', 'Accessible-height bed',
                 'Keypad', '24-hour check-in', 'Baby monitor',
                 'Laptop friendly workspace', 'Breakfast', 'Family/kid friendly',
                 'Free parking on street', 'Doorman', 'Carbon monoxide detector',
                 'Gym', 'Firm matress', 'First aid kit', 'Baby bath',
                 "Children’s dinnerware", 'Private entrance',
                 'Dishes and silverware', 'Game console', 'Step-free access',
                 'Hair dryer', 'Lock on bedroom door', 'Shampoo', 'Window guards',
                 'Bathtub', 'Pool', 'Babysitter recommendations',
                 'Private bathroom', 'Buzzer/wireless intercom',
                 'Wireless Internet', 'Extra pillows and blankets',
                 'Elevator in building', 'Accessible-height toilet', 'Dog(s)',
                 "Pack ’n Play/travel crib", 'Outlet covers', 'Washer', 'Hot tub',
                 'Wheelchair accessible', 'Dishwasher', 'Changing table',
                 'Free parking on premises', 'Long term stays allowed',
                 'Room-darkening shades', 'Cooking basics', 'Cable TV',
                 'translation missing: en.hosting_amenity_49', 'Iron',
                 'Patio or balcony', 'Wide clearance to bed',
                 'Flat smooth pathway to front door', 'Kitchen', 'Wide doorway',
                 'Doorman Entry', 'Wide hallway clearance', 'Stair gates',
                 'Garden or backyard', 'Pets live on this property', 'Safety card',
                 'Suitable for events', 'Grab-rails for shower and toilet',
                 'Path to entrance lit at night', 'BBQ grill', 'Cat(s)',
                 'Table corner guards', 'Essentials', 'Indoor fireplace',
                 'translation missing: en.hosting_amenity_50',
                 'Private living room', 'Internet', 'Luggage dropoff allowed',
                 'Other pet(s)', 'Oven', 'TV',
                 'Wide clearance to shower and toilet', 'Beach essentials',
                 'Hangers', 'Pets allowed', "Children’s books and toys"}
    amenities = sorted([amenity.title() for amenity in amenities if 'translation' not in amenity])

    cancellation_policy_types = sorted(['moderate', 'flexible', 'strict', 'no_refunds', 'super_strict_30',
                                        'long_term'])
    bed_types = sorted(['Real Bed', 'Airbed', 'Futon', 'Pull-out Sofa', 'Couch'])
    property_types = sorted(['House', 'Apartment', 'Loft', 'Dorm', 'Condominium',
                             'Bed & Breakfast', 'Other', 'Townhouse', 'Guesthouse', 'Boat',
                             'Hostel', 'Bungalow', 'Timeshare', 'Boutique hotel', 'Guest suite',
                             'Tent', 'In-law', 'Serviced apartment', 'Villa', 'Cabin',
                             'Earth House', 'Cave', 'Castle', 'Lighthouse', 'Hut',
                             'Vacation home', 'Chalet'])
    room_types = sorted(['Private room', 'Entire home/apt', 'Shared room'])
    return render_template('predict.html', amenities=amenities, cancellation_policy_types=cancellation_policy_types,
                           bed_types=bed_types, property_types=property_types, room_types=room_types)


@app.route("/process_form", methods=['POST'])
def process_form():
    venues = pd.read_csv("venues_new.csv")
    venue_types = ['courthouse', 'stadium', 'natural_feature', 'hospital', 'atm',
                   'cafe', 'store', 'jewelry_store', 'laundry', 'furniture_store',
                   'hindu_temple', 'aquarium', 'general_contractor', 'synagogue',
                   'bicycle_store', 'car_wash', 'car_dealer', 'florist',
                   'establishment', 'airport', 'beauty_salon', 'bakery', 'hair_care',
                   'movie_theater', 'locksmith', 'home_goods_store', 'moving_company',
                   'place_of_worship', 'plumber', 'meal_takeaway', 'health',
                   'roofing_contractor', 'local_government_office', 'physiotherapist',
                   'movie_rental', 'electrician', 'mosque', 'gas_station', 'spa',
                   'school', 'political', 'bar', 'electronics_store', 'night_club',
                   'library', 'point_of_interest', 'funeral_home', 'fire_station',
                   'taxi_stand', 'shoe_store', 'park', 'premise', 'pharmacy',
                   'amusement_park', 'light_rail_station', 'post_office', 'bank',
                   'convenience_store', 'rv_park', 'neighborhood', 'clothing_store',
                   'art_gallery', 'accounting', 'police', 'doctor', 'painter',
                   'university', 'restaurant', 'lodging', 'finance', 'meal_delivery',
                   'casino', 'car_repair', 'zoo', 'cemetery',
                   'grocery_or_supermarket', 'parking', 'embassy', 'museum',
                   'insurance_agency', 'campground', 'storage', 'department_store',
                   'book_store', 'hardware_store', 'subway_station',
                   'real_estate_agency', 'transit_station', 'train_station',
                   'subpremise', 'bowling_alley', 'gym', 'pet_store',
                   'veterinary_care', 'dentist', 'food', 'bus_station', 'car_rental',
                   'travel_agency', 'shopping_mall', 'lawyer', 'church',
                   'liquor_store']
    for row in venues['types']:
        venue_types += [venue_type.replace("'", '') for venue_type in row[2:-2].split(', ')]
    venue_types = list(set(venue_types))

    def get_amenities(long_, lat, max_distance):
        venues_ = venues.copy()
        delta = 0.00098621649 * 180 * max_distance / np.pi

        venues_ = venues_[venues_['longitude'] < long_ + delta]
        venues_ = venues_[venues_['longitude'] > long_ - delta]

        venues_ = venues_[venues_['latitude'] < lat + delta]
        venues_ = venues_[venues_['latitude'] > lat - delta]

        return venues_[venue_types].sum()

    data = {}
    data['name'] = request.form.get("name")
    data['accommodates'] = request.form.get("accommodates")  # Need to add
    data['bathrooms'] = request.form.get("bathrooms")
    data['bedrooms'] = request.form.get("bedrooms")

    for bed_type in ['Real Bed', 'Airbed', 'Futon', 'Pull-out Sofa', 'Couch']:
        if bed_type == request.form.get('bed_type'):
            data["bed_type_" + bed_type] = 1
        else:
            data["bed_type_" + bed_type] = 0

    data['beds'] = request.form.get("beds")

    for cancellation_policy in ['moderate', 'flexible', 'strict', 'no_refunds', 'super_strict_30',
                                'long_term']:
        if cancellation_policy == request.form.get("cancellation_policy"):
            data["cancellation_policy_" + cancellation_policy] = 1
        else:
            data["cancellation_policy_" + cancellation_policy] = 0

    data['city'] = str(request.form.get("city")) # Need to add
    data['instant_bookable'] = False
    data['latitude'] = str(request.form.get("latitude"))
    data['longitude'] = str(request.form.get("longitude"))
    print(data['latitude'], data['longitude'])

    geolocator = Nominatim(user_agent="specify_your_app_name_here")
    location = geolocator.reverse(data['latitude'] + ", " + data['longitude'])
    print(location)
    zipcode_str = location.address
    zipcode_str = zipcode_str[-31:-26]

    data['metropolitan'] = str(request.form.get("metropolitan"))

    for property_type in ['House', 'Apartment', 'Loft', 'Dorm', 'Condominium',
                          'Bed & Breakfast', 'Other', 'Townhouse', 'Guesthouse', 'Boat',
                          'Hostel', 'Bungalow', 'Timeshare', 'Boutique hotel', 'Guest suite',
                          'Tent', 'In-law', 'Serviced apartment', 'Villa', 'Cabin',
                          'Earth House', 'Cave', 'Castle', 'Lighthouse', 'Hut',
                          'Vacation home', 'Chalet']:
        if property_type == request.form.get("property_type"):
            data["property_type_" + property_type] = 1
        else:
            data["property_type_" + property_type] = 0

    for room_type in ['Private room', 'Entire home/apt', 'Shared room']:
        if room_type == request.form.get("room_type"):
            data["room_type_" + room_type] = 1
        else:
            data["room_type_" + room_type] = 0

    data['zipcode'] = int(zipcode_str)
    data['discount'] = request.form.get("discount")

    data['state'] = 'NA'

    amenities = {'24-hour check-in',
                 'Accessible-height bed',
                 'Accessible-height toilet',
                 'Air conditioning',
                 'BBQ grill',
                 'Baby bath',
                 'Baby monitor',
                 'Babysitter recommendations',
                 'Bathtub',
                 'Beach essentials',
                 'Bed linens',
                 'Breakfast',
                 'Buzzer/wireless intercom',
                 'Cable TV',
                 'Carbon monoxide detector',
                 'Cat(s)',
                 'Changing table',
                 "Children’s books and toys",
                 "Children’s dinnerware",
                 'Cleaning before checkout',
                 'Coffee maker',
                 'Cooking basics',
                 'Crib',
                 'Dishes and silverware',
                 'Dishwasher',
                 'Dog(s)',
                 'Doorman',
                 'Doorman Entry',
                 'Dryer',
                 'Elevator in building',
                 'Essentials',
                 'Ethernet connection',
                 'Extra pillows and blankets',
                 'Family/kid friendly',
                 'Fire extinguisher',
                 'Fireplace guards',
                 'Firm matress',
                 'First aid kit',
                 'Flat smooth pathway to front door',
                 'Free parking on premises',
                 'Free parking on street',
                 'Game console',
                 'Garden or backyard',
                 'Grab-rails for shower and toilet',
                 'Gym',
                 'Hair dryer',
                 'Hangers',
                 'Heating',
                 'High chair',
                 'Hot tub',
                 'Hot water',
                 'Indoor fireplace',
                 'Internet',
                 'Iron',
                 'Keypad',
                 'Kitchen',
                 'Laptop friendly workspace',
                 'Lock on bedroom door',
                 'Lockbox',
                 'Long term stays allowed',
                 'Luggage dropoff allowed',
                 'Microwave',
                 'Other pet(s)',
                 'Outlet covers',
                 'Oven',
                 "Pack ’n Play/travel crib",
                 'Path to entrance lit at night',
                 'Patio or balcony',
                 'Pets allowed',
                 'Pets live on this property',
                 'Pool',
                 'Private bathroom',
                 'Private entrance',
                 'Private living room',
                 'Refrigerator',
                 'Room-darkening shades',
                 'Safety card',
                 'Self Check-In',
                 'Shampoo',
                 'Smartlock',
                 'Smoke detector',
                 'Smoking allowed',
                 'Stair gates',
                 'Step-free access',
                 'Stove',
                 'Suitable for events',
                 'TV',
                 'Table corner guards',
                 'Washer',
                 'Washer / Dryer',
                 'Wheelchair accessible',
                 'Wide clearance to bed',
                 'Wide clearance to shower and toilet',
                 'Wide doorway',
                 'Wide hallway clearance',
                 'Window guards',
                 'Wireless Internet',
                 'translation missing: en.hosting_amenity_49',
                 'translation missing: en.hosting_amenity_50'}

    '''
    amenities = sorted([amenity.title() for amenity in amenities if 'translation' not in amenity])
    amenity_str = []

    for i in range(0, len(amenities)):
        amenity_flag = request.form.get(amenities[i])
        if amenity_flag:
            amenity_str.append(amenities[i])
    
    data['amenities'] = amenity_str
    '''
    for amenity in amenities:
        if request.form.get(amenity):
            data[amenity] = True
        else:
            data[amenity] = False

    demographics = pd.read_csv("demographics.csv")
    print(demographics.head())

    data = pd.DataFrame({key: [data[key]] for key in data})
    for venue_type in venue_types:
        data[venue_type] = np.nan

    data = data.merge(demographics, left_on='zipcode', right_on="zipcode")
    print(data)
    data[venue_types] = data[['longitude', 'latitude']].apply(
        lambda x: get_amenities(float(x.longitude), float(x.latitude), 0.1), axis=1)

    print(dict(data.iloc[0]).keys())

    print("Preprocessing is done")

    with open("tfidf.pkl", "rb") as f:
        tfidf = pkl.load(f)

    english_stopwords = stopwords.words("english")
    lemm = WordNetLemmatizer()

    def preprocess_text(text):
        tokens = [lemm.lemmatize(word) for word in word_tokenize(str(text).lower())]
        tokens = [token for token in tokens if token not in english_stopwords \
                  and token != " " \
                  and token.strip() not in punctuation
                  and not token.isdigit()]

        text = " ".join(tokens)

        return text

    data['names_preproc'] = data['name'].map(preprocess_text)
    transformed_names = tfidf.transform(data['names_preproc'])

    for i in range(200):
        data[str(i)] = transformed_names[:, i].todense()

    new_columns = ['accommodates', 'bathrooms', 'bedrooms', 'beds',
     'instant_bookable', 'latitude', 'longitude', 'zipcode',
     'High chair', 'Smartlock', 'Fireplace guards', 'Heating',
     'Cleaning before checkout', 'Smoke detector', 'Lockbox',
     'Refrigerator', 'Hot water', 'Dryer', 'Self Check-In',
     'Smoking allowed', 'Stove', 'Bed linens', 'Microwave', 'Crib',
     'Washer / Dryer', 'Air conditioning', 'Fire extinguisher',
     'Coffee maker', 'Ethernet connection', 'Accessible-height bed',
     'Keypad', '24-hour check-in', 'Baby monitor',
     'Laptop friendly workspace', 'Breakfast', 'Family/kid friendly',
     'Free parking on street', 'Doorman', 'Carbon monoxide detector',
     'Gym', 'Firm matress', 'First aid kit', 'Baby bath',
     "Children’s dinnerware", 'Private entrance',
                                               'Dishes and silverware', 'Game console', 'Step-free access',
     'Hair dryer', 'Lock on bedroom door', 'Shampoo', 'Window guards',
     'Bathtub', 'Pool', 'Babysitter recommendations',
     'Private bathroom', 'Buzzer/wireless intercom',
     'Wireless Internet', 'Extra pillows and blankets',
     'Elevator in building', 'Accessible-height toilet', 'Dog(s)',
     "Pack ’n Play/travel crib", 'Outlet covers', 'Washer', 'Hot tub',
     'Wheelchair accessible', 'Dishwasher',
     'Changing table',
     'Free parking on premises', 'Long term stays allowed',
     'Room-darkening shades', 'Cooking basics', 'Cable TV',
     'translation missing: en.hosting_amenity_49', 'Iron',
     'Patio or balcony', 'Wide clearance to bed',
     'Flat smooth pathway to front door', 'Kitchen', 'Wide doorway',
     'Doorman Entry', 'Wide hallway clearance', 'Stair gates',
     'Garden or backyard', 'Pets live on this property', 'Safety card',
     'Suitable for events', 'Grab-rails for shower and toilet',
     'Path to entrance lit at night', 'BBQ grill', 'Cat(s)',
     'Table corner guards', 'Essentials', 'Indoor fireplace',
     'translation missing: en.hosting_amenity_50',
     'Private living room', 'Internet', 'Luggage dropoff allowed',
     'Other pet(s)', 'Oven', 'TV',
     'Wide clearance to shower and toilet', 'Beach essentials',
     'Hangers', 'Pets allowed', "Children’s books and toys", "discount",
    'cancellation_policy_flexible',
     'cancellation_policy_long_term',
     'cancellation_policy_moderate', 'cancellation_policy_no_refunds',
     'cancellation_policy_strict',
     'cancellation_policy_super_strict_30', 'bed_type_Airbed',
     'bed_type_Couch', 'bed_type_Futon', 'bed_type_Pull-out Sofa',
     'bed_type_Real Bed', 'room_type_Entire home/apt',
     'room_type_Private room', 'room_type_Shared room',
     'property_type_Apartment', 'property_type_Bed & Breakfast',
     'property_type_Boat', 'property_type_Boutique hotel',
     'property_type_Bungalow', 'property_type_Cabin',
     'property_type_Castle', 'property_type_Cave',
     'property_type_Chalet', 'property_type_Condominium',
     'property_type_Dorm', 'property_type_Earth House',
     'property_type_Guest suite', 'property_type_Guesthouse',
     'property_type_Hostel', 'property_type_House', 'property_type_Hut',
     'property_type_In-law', 'property_type_Lighthouse',
     'property_type_Loft', 'property_type_Other',
     'property_type_Serviced apartment', 'property_type_Tent',
     'property_type_Timeshare', 'property_type_Townhouse',
     'property_type_Vacation home', 'property_type_Villa',
     '5_years_or_less', '5-9_years', '10-14_years', '15-19_years',
     '20-24_years', '25-34_years', '35-44_years', '45-54_years',
     '55-59_years', '60-64_years', '65-74_years', '75-84_years',
     '85_years_or_more', 'households', '$9,999_or_less',
     '$10,000-$14,999', '$15,000-$24,999', '$25,000-$34,999',
     '$35,000-$49,999', '$50,000-$64,999', '$65,000-$74,999',
     '$75,000-$99,999', '$100,000_or_more', 'median_household_income',
     'mean_household_income', 'lawyer', 'light_rail_station',
     'locksmith', 'subpremise', 'roofing_contractor',
     'point_of_interest', 'plumber', 'health', 'premise',
     'fire_station', 'police', 'food', 'natural_feature',
     'subway_station', 'department_store', 'cemetery', 'book_store',
     'beauty_salon', 'liquor_store', 'furniture_store',
     'physiotherapist', 'car_wash', 'neighborhood', 'jewelry_store',
     'bowling_alley', 'movie_rental', 'aquarium', 'library',
     'meal_delivery', 'gas_station', 'bicycle_store', 'taxi_stand',
     'mosque', 'night_club', 'accounting', 'convenience_store',
     'hair_care', 'shoe_store', 'museum', 'bus_station', 'store',
     'travel_agency', 'movie_theater', 'hospital', 'electrician',
     'cafe', 'political', 'airport', 'car_rental', 'car_repair',
     'university', 'grocery_or_supermarket', 'veterinary_care',
     'laundry', 'funeral_home', 'hindu_temple', 'gym', 'post_office',
     'pharmacy', 'parking', 'school', 'stadium', 'bar', 'zoo', 'church',
     'local_government_office', 'finance', 'art_gallery',
     'real_estate_agency', 'place_of_worship', 'hardware_store',
     'florist', 'clothing_store', 'painter', 'bank', 'meal_takeaway',
     'spa', 'home_goods_store', 'restaurant', 'bakery', 'storage',
     'doctor', 'synagogue', 'transit_station', 'campground', 'lodging',
     'courthouse', 'rv_park', 'embassy', 'casino', 'moving_company',
     'amusement_park', 'pet_store', 'insurance_agency', 'park',
     'shopping_mall', 'dentist', 'electronics_store', 'atm',
     'establishment', 'car_dealer', 'train_station',
     'general_contractor', '0', '1', '2', '3', '4', '5', '6', '7', '8',
     '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19',
     '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30',
     '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41',
     '42', '43', '44', '45', '46', '47', '48', '49', '50', '51', '52',
     '53', '54', '55', '56', '57', '58', '59', '60', '61', '62', '63',
     '64', '65', '66', '67', '68', '69', '70', '71', '72', '73', '74',
     '75', '76', '77', '78', '79', '80', '81', '82', '83', '84', '85',
     '86', '87', '88', '89', '90', '91', '92', '93', '94', '95', '96',
     '97', '98', '99', '100', '101', '102', '103', '104', '105', '106',
     '107', '108', '109', '110', '111', '112', '113', '114', '115',
     '116', '117', '118', '119', '120', '121', '122', '123', '124',
     '125', '126', '127', '128', '129', '130', '131', '132', '133',
     '134', '135', '136', '137', '138', '139', '140', '141', '142',
     '143', '144', '145', '146', '147', '148', '149', '150', '151',
     '152', '153', '154', '155', '156', '157', '158', '159', '160',
     '161', '162', '163', '164', '165', '166', '167', '168', '169',
     '170', '171', '172', '173', '174', '175', '176', '177', '178',
     '179', '180', '181', '182', '183', '184', '185', '186', '187',
     '188', '189', '190', '191', '192', '193', '194', '195', '196',
     '197', '198', '199', 'city_arverne', 'city_astoria',
     'city_auburndale', 'city_averne', 'city_bay ridge', 'city_bayside',
     'city_bedstuy', 'city_belle harbor', 'city_bellerose', 'city_bk',
     'city_briarwood', 'city_bronx ny', 'city_brooklyn',
     'city_bushwick', 'city_cambria heights', 'city_carroll gardens',
     'city_college point', 'city_corona', 'city_east elmhurst',
     'city_east williamsburg', 'city_elmhurst', 'city_elmont',
     'city_elmuhrust', 'city_far rockaway', 'city_floral park',
     'city_flushing', 'city_forest hills', 'city_fort greene',
     'city_fresh meadows', 'city_glendale', 'city_glendale\nglendale',
     'city_greenpoint', 'city_harlem', "city_hell's kitchen",
     'city_hollis', 'city_howard beach', 'city_jackson heights',
     'city_jamaica', 'city_kew gardens', 'city_kew gardens hills',
     'city_kips bay', 'city_l.i.c', 'city_laurelton', 'city_lawrence',
     'city_lic', 'city_long island city', 'city_longislandcity',
     'city_lower east side', 'city_manhattan', 'city_manhattan ny',
     'city_maspeth', 'city_middle village', 'city_new york', 'city_nyc',
     'city_oakland gardens', 'city_ozone park', 'city_park slope',
     'city_parkchester', 'city_parkchester bronx', 'city_queens',
     'city_red hook', 'city_rego park', 'city_richmond hill',
     'city_ridgewood', 'city_riverdale', 'city_rockaway beach',
     'city_rosedale', 'city_saint albans', 'city_south ozone park',
     'city_south richmond hill', 'city_sprinfield gardens',
     'city_springfield gardens', 'city_st albans', 'city_st. albans',
     'city_staten island', 'city_statenisland', 'city_sunnysidebronx',
     'city_valley stream', 'city_wadsworth terrace', 'city_whitestone',
     'city_williamsburg', 'city_woodside', 'city_yonkers',
     'metropolitan_NYC', 'metropolitan_dc', 'state_NY']

    for new_column in new_columns:
        if new_column not in data.columns:
            data[new_column] = np.nan

    data['city_'+data['city']] = [1]
    data['state_NY'] = [1]
    data.fillna(0, inplace=True)

    data.drop(columns = ['name','city','metropolitan','state', 'names_preproc'], inplace = True)

    for column in data.columns:
        if column not in new_columns:
            print(column)

    with open("model_200.pkl", "rb") as f:
        model = pkl.load(f)

    data[['accommodates', 'bathrooms', 'bedrooms', 'beds', 'latitude', 'longitude', 'discount',
    '$10,000-$14,999', '$15,000-$24,999', '$25,000-$34,999', '$35,000-$49,999', '$50,000-$64,999', '$65,000-$74,999',
    '$75,000-$99,999', '$100,000_or_more', 'median_household_income', 'mean_household_income']] = data[['accommodates', 'bathrooms', 'bedrooms', 'beds', 'latitude', 'longitude', 'discount',
    '$10,000-$14,999', '$15,000-$24,999', '$25,000-$34,999', '$35,000-$49,999', '$50,000-$64,999', '$65,000-$74,999',
    '$75,000-$99,999', '$100,000_or_more', 'median_household_income', 'mean_household_income']].astype(float)

    with open("columns_train.pkl", 'rb') as f:
        columns_s = pkl.load(f)

    price = model.predict(data[columns_s])[0]
    print(price)
    prop_price = int(request.form.get('price'))
    print(prop_price)
    print(1/(1-(prop_price*0.02/(price*12*30))))
    pay_back = int(np.log(1/(1-(prop_price*0.02/(price*12*30))))/np.log(1+0.02))
    return render_template('price.html', price=price, price_div  = int(0.97*price), prop_price = prop_price,
                           pay_back=pay_back)
