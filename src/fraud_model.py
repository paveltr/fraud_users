from libs.env import *


logger = logging.getLogger('predict_fraudsters.'+__name__)
logging.basicConfig(filename='logs/fraud_detection.log', level=logging.DEBUG, 
                    format='%(asctime)s %(levelname)s %(name)s %(message)s')


def download_files(bucket_name, user_path, tracks_path, streams_path, data_path = r'data',
                              users_name = 'users', tracks_name = 'tracks', streams_name = 'streams'):
    """
    Funtion downloads files from specified Google Storage
    
    Args:
    bucket_name : string
        google cloud bucket url
    user_path : string
        path to file with users data on bucket
    tracks_path : string
        path to file with tracks data on bucket
    streams_path : string
        path to file with streams data on bucket
    data_path : string
        local folder to strore files
    users_name : string
        local filename for file with users
    tracks_name : string
        local filename for file with tracks
    streams_name : string
        local filename for file with streams

    Returns:
        None

    Raises:
        Errors with downloading or saving files
    """
    try:
        os.system(r'gsutil cp {0}/{1} {2}/{3}'.format(bucket_name, user_path, data_path, users_name))
        os.system(r'gsutil cp {0}/{1} {2}/{3}'.format(bucket_name, tracks_path, data_path, tracks_name))
        os.system(r'gsutil cp {0}/{1} {2}/{3}'.format(bucket_name, streams_path, data_path, streams_name))
        logger.info('Files downloaded')
    except Exception as e:
        logger.error('File downloading process failed with error: %s' % e)
        
def upload_files(bucket_name, filename = 'fraud_users', data_path = r'data'):
    """
    Funtion uploads file with fraud users to specified Google Storage
    
    Args:
    bucket_name : string
        google cloud bucket url
    filename : string
        local name of file with fraud users
    data_path : string
        local folder to strore files

    Returns:
        None

    Raises:
        Errors with with saving and uploading file
    """
    try:
        result_file = str(datetime.datetime.today())[:10].replace('-', '') + '_' + filename
        os.system(r'gsutil cp {2}/{3} {0}/{1}'.format(bucket_name, result_file, data_path, filename))
        logger.info('Result file uploaded to Google Cloud')
    except Exception as e:
        logger.error('File uploading process failed with error: %s' % e)
    

def read_files(data_path, users_name = 'users', tracks_name = 'tracks', streams_name = 'streams'):
    """
    Function reads users, tracks and streams data fro, local folder
    
    Args:
    data_path : string
        local folder to strore files
    users_name : string
        local name of file with users
    tracks_name : string
        local name of file with tracks
    streams : string
        local name of file with tracks
    
    Returns:
        pandas dataframes

    Raises:
        Errors with with saving and uploading file
    """
    
    logger.info('Files processing started')
    
    # users
    users = pd.read_csv(data_path + '/' + users_name, compression = 'gzip', header = None)
    users.columns = ['access', 'birth_year', 'country', 'gender', 'user_id']
    users['user_id'] = users['user_id'].map(get_id)
    users['birth_year'] = users['birth_year'].map(lambda x: clean_text(x , 'numbers')).replace('', '2000').astype(int)
    users['age'] = datetime.datetime.today().date().year - users['birth_year']
    # tracks
    tracks=gzip.open(data_path + '/' + tracks_name,'rb')
    tracks=tracks.read().decode().split('\n')
    strings = []

    for track in tracks:    
        line = json.loads(track)
        _ = []
        for key, value in line.items():
            _.append(key  + ' : ' + value)
        strings.append(_)

    tracks = pd.DataFrame(strings)
    tracks.columns = ['album_artist', 'album_code', 'album_name', 'track_id', 'track_name']
    tracks['track_id'] = tracks['track_id'].map(get_id)
    
    # streams
    streams = pd.read_csv(data_path + '/' + streams_name, compression = 'gzip', header = None)
    streams.columns = ['device_type', 'length', 'os', 'timestamp', 'track_id', 'user_id']
    streams['user_id'] = streams['user_id'].map(get_id)
    streams['track_id'] = streams['track_id'].map(get_id)
    streams['length'] = streams['length'].map(lambda x: clean_text(x , 'numbers')).astype(int)
    streams = streams.sort_values(by = ['user_id', 'timestamp'])
    streams['timestamp'] = streams['timestamp'].map(lambda x: clean_text(x , 'numbers')[:10]).astype(int)
    streams['timestamp_previous'] = streams.groupby('user_id', sort = False)['timestamp'].shift(1)
    streams['gap'] = streams['timestamp'] - streams['timestamp_previous']
    streams['gap'] = streams['gap'] / (60)
    
    streams = pd.merge(streams, tracks[['track_id', 'album_artist', 'album_code']],
                        how = 'left', on = 'track_id')
    
    logger.info('Files processed')
    
    return users, tracks, streams

    
def find_outliers(users, tracks, streams, encoders_path, train, std_ratio):
     """
    Function detects users with extreme activity
    
    Args:
    users : pandas dataframe
        pandas dataframe with users
    tracks : pandas dataframe
        pandas dataframe with tracks
    streams : pandas dataframe
        pandas dataframe with streams
    encoders_path : string
        local path to data encoder
    train : boolean
        option to choose either we need to train fraud mode on current data or not
    std_ratio : float
        threshold in standard deviation scale to select outliers
        
    Returns:
        pandas dataframe

    Raises:
        Errors with pandas transformations or encoder existence
    """
        
    logger.info('Outliers seach process started')
 
    if not train:
        encoder = load_pickle(r'encoder', encoders_path)
    else:
        encoder = {}
        
    allusers = users[['user_id', 'gender', 'age', 'access', 'country']]
    allusers['gender'] = (allusers['gender'] == 'gender:"female"')*1
    
    top_countries = encoder['top_countries'] if not train else users['country'].value_counts().index[:4].tolist()
    
    if train: encoder['top_countries'] = [t for t in top_countries]
    
    for i, t in enumerate(top_countries):
        allusers['top_country_' + str(i)] = (allusers['country'] == t)*1
    allusers.drop(['country'], axis = 1, inplace = True)

    for i, t in enumerate(['{"access":"free"', '{"access":"premium"']):
        allusers['access_' + str(i)] = (allusers['access'] == t)*1
    allusers.drop(['access'], axis = 1, inplace = True)

    allusers['user_track_unique'] = allusers['user_id'].map(streams.groupby('user_id')['track_id'].nunique())

    allusers['user_stream_length_mean'] = allusers['user_id'].map(streams.groupby('user_id')['length'].mean())

    allusers['user_gap_mean'] = \
    allusers['user_id'].map(streams[streams['gap'].notnull()].groupby('user_id')['gap'].mean()).fillna(0)

    allusers['user_track_count'] = allusers['user_id'].map(streams.groupby('user_id')['track_id'].count())

    track_stat = streams['track_id'].value_counts().reset_index()
    track_stat.columns = ['track_id', 'total_count']
    user_track = streams.groupby(['user_id', 'track_id'])['device_type'].count().reset_index()\
    .rename(columns = {'device_type' : 'user_count'})
    user_track = pd.merge(user_track, track_stat, how = 'inner', on = 'track_id')

    user_track['user_share'] = user_track['user_count'] / user_track['total_count']

    allusers['user_track_mean_share'] = allusers['user_id'].map(user_track.groupby('user_id')['user_share'].mean())
    allusers['n_albums'] = allusers['user_id'].map(streams.groupby('user_id')['album_code'].nunique().to_dict())
    allusers['n_artists'] = allusers['user_id'].map(streams.groupby('user_id')['album_artist'].nunique().to_dict())
    
    if train:
        for col in allusers.columns[9:]:
            mean = allusers[col].mean()
            std = allusers[col].std()
            allusers[col] = (allusers[col] - mean) / std
            encoder[col] = (mean, std)
    else:
        for col in allusers.columns[9:]:
            mean = encoder[col][0]
            std = encoder[col][1]
            allusers[col] = (allusers[col] - mean) / std
        
    if train:
        clf = IsolationForest(contamination=0.3, n_jobs = -1, random_state = 0, 
                              max_features = allusers.iloc[:, 1:].shape[1])
    else:
        clf = load_pickle(r'IsolationForest', encoders_path)
        
    allusers['outlier'] = (clf.predict(allusers.iloc[:, 1:]) == -1)*1
    
    allusers['outlier'] = ((abs(allusers['user_track_unique']) > std_ratio) & \
                       (abs(allusers['user_track_count']) > std_ratio) & (allusers['outlier'] == 1))*1
    
    if train: save_pickle(encoder, r'encoder', encoders_path)
        
    logger.info('Outliers successfully found')
    
    return allusers[allusers['outlier'] == 1]


def find_similar(streams, allusers, radius, n_duplicates):
    """
    Function detects neighbors of the point within specified radius
    
    Args:
    streams : pandas dataframe
        pandas dataframe with streams
    allusers : pandas dataframe
        pandas dataframe with suspicious users
    radius : float
        neighborhood radius at which we look for outliers
    n_duplicates : integer
        threshold to select users with specific number of possible duplicted accounts
        
    Returns:
        pandas dataframe

    Raises:
        Errors with pandas transformations
    """
    
    logger.info('Find_similar started')
    
    only_suspicious = streams[streams['user_id'].isin(allusers[allusers['outlier'] == 1]['user_id'].unique())]
    only_suspicious = only_suspicious.groupby('user_id')['track_id'].apply(list).reset_index()

    tv = TfidfVectorizer()
    tv.fit([' '.join(ll for ll in l).strip() for l in only_suspicious['track_id'].values])

    features_vec = tv.transform([' '.join(ll for ll in l).strip() for l in only_suspicious['track_id'].values])
    lshf = LSHForest(random_state=42, n_candidates=100, n_estimators=10, n_neighbors=5,
     radius=0.1)
    
    lshf.fit(features_vec)
    
    distances = lshf.radius_neighbors(features_vec, radius = radius)
    index_match = only_suspicious.set_index(only_suspicious.index)['user_id'].to_dict()
    
    neigbors = find_neighbors(distances, index_match)
    only_suspicious['possible_duplicate_accounts'] = neigbors
    only_suspicious['n_duplicates'] = only_suspicious['possible_duplicate_accounts'].map(lambda x: len(x))
    
    logger.info('Find similar finished')
    
    return only_suspicious[only_suspicious['n_duplicates'] >= n_duplicates]
    

def check_distance(connected, streams):
    """
    Function checks that users with possible duplicated accounts have specified number of duplicates
    
    Args:
    connected : pandas dataframe
        pandas dataframe with users
    streams : pandas dataframe
        pandas dataframe with streams
        
    Returns:
        list

    Raises:
        Errors with pandas transformations
    """
    
    logger.info('Distance check started')
    
    left = [] 
    right = []

    for row in connected.iterrows():
        for con in row[1]['possible_duplicate_accounts']:
            pair = (row[1]['user_id'], con) if row[1]['user_id'] < con else (con, row[1]['user_id'])
            left.append(pair[0])
            right.append(pair[1])
            
    connected = pd.DataFrame({'left' : left, 'right' : right}).drop_duplicates(keep = 'first')
    
    track_left = pd.merge(connected[['left', 'right']], 
                      streams[['user_id', 'track_id']], how = 'inner', left_on = 'left', right_on = 'user_id')

    track_right = pd.merge(connected[['left', 'right']], 
                          streams[['user_id', 'track_id']], how = 'inner', left_on = 'right', right_on = 'user_id')

    tracks = pd.merge(track_left, track_right, how = 'inner', on = ['left', 'right', 'track_id'])

    tracks = tracks.groupby(['left', 'right'])['track_id'].nunique().reset_index().rename(columns = {'track_id' : 'n_tracks'})
    connected = pd.merge(connected, tracks, how = 'left', on = ['left', 'right'])
    connected['left_tracks'] = connected['left'].map(streams.groupby('user_id')['track_id'].nunique().to_dict())
    connected['right_tracks'] = connected['right'].map(streams.groupby('user_id')['track_id'].nunique().to_dict())
    connected['similarity'] = connected['n_tracks'] / (connected['left_tracks']  + connected['right_tracks']) 
    
    logger.info('Distance check finished')
    
    return connected[connected['similarity'] > 0.4]['left'].unique().tolist() + \
           connected[connected['similarity'] > 0.4]['right'].unique().tolist() 

def create_df(df):
    """
    Function checks consistence of dataframe columns
    
    Args:
    df : pandas dataframe
        pandas dataframe with users
        
    Returns:
        pandas dataframe

    Raises:
        Errors with pandas transformations
    """
    cols = [
    'user_id', 'gender', 'age', 'user_track_unique', 
    'user_stream_length_mean', 'user_gap_mean',
    'user_track_count', 'user_track_mean_share', 
    'n_albums', 'n_artists', 'possible_duplicate_accounts', 
    'n_duplicates', 'fraud_type']
    for col in cols:
        if col not in df.columns:
            df[col] = np.nan
    return df[cols]

def merge_df(allusers, connected):
    """
    Function merges dataframe with suspicious users and users with possible multiple accounts
    
    Args:
    allusers : pandas dataframe
        pandas dataframe with users
    connected : pandas dataframe
        pandas dataframe with users
        
    Returns:
        pandas dataframe

    Raises:
        Errors with pandas transformations
    """
    result = pd.merge(allusers, connected, how = 'left', on = 'user_id')
    result['fraud_type'] = result['possible_duplicate_accounts']\
    .apply(lambda x: 'Type I (extreme activity)' if type(x) != list else 'Type II ( multiple accounts)')
    logger.info('Files merged')
    return create_df(result)
    
def find_fraud(train, radius, encoders_path, data_path, std_ratio, n_duplicates):
    """
    Function finds fraud users in data
    
    Args:
    train : boolean
        option to choose either we need to train fraud mode on current data or not
    radius : float
        neighborhood radius at which we look for outliers
    encoders_path : string
        local path to data encoder
    data_path : string
        local folder to strore files
    std_ratio : float
        threshold in standard deviation scale to select outliers
    n_duplicates : integer
        threshold to select users with specific number of possible duplicted accounts
        
    Returns:
        pandas dataframe

    Raises:
        Errors with pandas transformations
    """
    logger.info('Fraud search started')
    
    users, tracks, streams = read_files(data_path)
    
    allusers = find_outliers(users, tracks, streams, encoders_path, train, std_ratio)
    
    if allusers.shape[0] == 0: return create_df(all_users)
    
    del users, tracks
    streams = streams[streams['user_id'].isin(allusers['user_id'].unique())]
    
    multiple_accounts = find_similar(streams, allusers, radius, n_duplicates)

    if multiple_accounts.shape[0] == 0: return create_df(all_users)
    
    connected = check_distance(multiple_accounts, streams)
        
    if len(connected) == 0: return create_df(all_users)
    
    logger.info('Fraud search finished')
    
    return merge_df(allusers, multiple_accounts[multiple_accounts['user_id'].isin(connected)])

def save_fraud(train, radius, encoders_path, data_path, std_ratio, n_duplicates):
    """
    Function finds fraud users in data
    
    Args:
    train : boolean
        option to choose either we need to train fraud mode on current data or not
    radius : float
        neighborhood radius at which we look for outliers
    encoders_path : string
        local path to data encoder
    data_path : string
        local folder to strore files
    std_ratio : float
        threshold in standard deviation scale to select outliers
    n_duplicates : integer
        threshold to select users with specific number of possible duplicted accounts
        
    Returns:
        pandas dataframe

    Raises:
        Errors with pandas transformations
    """
    df = find_fraud(train, radius, encoders_path, data_path, std_ratio, n_duplicates)
    df.to_csv(data_path + '/' + 'fraud_users', index = False, compression = 'gzip')
    logger.info('File with fraud users saved')
    
def main(bucket_name, user_path, tracks_path, streams_path, 
         train = False, encoders_path = r'encoders', 
         data_path = r'data', 
         std_ratio = 2, n_duplicates = 5, radius = 0.01):
    """
    Fraud model main function
    
    Args:
    bucket_name : string
        google cloud bucket url
    user_path : string
        path to file with users data on bucket
    tracks_path : string
        path to file with tracks data on bucket
    streams_path : string
        path to file with streams data on bucket
    train : boolean
        option to choose either we need to train fraud mode on current data or not
    encoders_path : string
        local path to data encoder
    data_path : string
        local folder to strore files
    std_ratio : float
        threshold in standard deviation scale to select outliers
    n_duplicates : integer
        threshold to select users with specific number of possible duplicted accounts
    radius : float
        neighborhood radius at which we look for outliers
        
    Returns:
        pandas dataframe

    Raises:
        Errors with pandas transformations
    """
    try:
        download_files(bucket_name, user_path, tracks_path, streams_path)
        save_fraud(train, radius, encoders_path, data_path, std_ratio, n_duplicates)
        upload_files(bucket_name)
    except Exception as e:
        logger.error('Script failed with error: %s' % e)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='fraud_app')
    parser.add_argument('--b_name', action="store", dest='b_name', default='')
    parser.add_argument('--u_path', action="store", dest='u_path', default='')
    parser.add_argument('--t_path', action="store", dest='t_path', default='')
    parser.add_argument('--s_path', action="store", dest='s_path', default='')
    parser.add_argument('--std_ratio', action="store", dest='std_ratio', default=2)
    parser.add_argument('--n_duplicates', action="store", dest='n_duplicates', default=5)
    parser.add_argument('--radius', action="store", dest='radius', default=0.01)
    parser.add_argument('--train', action="store", dest='train', default=False)
    
    args = parser.parse_args()
    
    if args.b_name == '' or args.b_name == '' or args.b_name == '' or args.b_name == '':
        raise Exception('Check paths to bucket and data file to be specified!')
        
    main(args.b_name, args.u_path, args.t_path, args.s_path, 
                     train = args.train, std_ratio = args.std_ratio, 
                     n_duplicates = args.n_duplicates, radius = args.radius)