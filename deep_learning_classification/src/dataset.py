import pandas as pd
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from src.utils.functions import createGrams, mkdir_if_not_exists, verify_existence_data
from src.utils.cat_ipinfo import CatIPInformation
from category_encoders.hashing import HashingEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers.experimental import preprocessing
from sklearn.preprocessing import LabelEncoder

class Darknet:
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = pd.read_csv(f"{self.data_path}/Darknet.csv", low_memory=False)
        darknet_processing = DarknetDataProcessing(self.data, self.data_path)
        if not verify_existence_data(f"{self.data_path}/processed/darknet_dataset_processed_encoded.csv"):
            darknet_processing.doPreProcessing()
            self.samples, self.model_samples = darknet_processing.getProcessedData()
        else:
            self.samples, self.model_samples = pd.read_csv(f"{self.data_path}/processed/darknet_dataset_processed.csv", low_memory=False), pd.read_csv(f"{self.data_path}/processed/darknet_dataset_processed_encoded.csv", low_memory=False)

    def exportProcessedData(self):
        mkdir_if_not_exists(f"{self.data_path}/processed")
        self.samples.to_csv(f"{self.data_path}/processed/darknet_dataset_processed.csv", index=False)
        self.model_samples.to_csv(f"{self.data_path}/processed/darknet_dataset_processed_encoded.csv", index=False)

    def getTrainTestValData(self, size_test: float, job: bool) -> pd.DataFrame:
        #job: True classification
        #job: False characterization
        model_data = self.model_samples.copy()
        self.job = job

        if self.job:
            del model_data['Label.1']
            model_data['target'] = np.where(model_data['Label'] == 'Darknet', 0, 1)
        else:
            del model_data['Label']
            model_data['target'] = model_data['Label.1']
            encoder = LabelEncoder()
            encoder.fit(model_data['target'])
            model_data['target'] = encoder.transform(model_data['target'])


        self.train, self.test = train_test_split(model_data, test_size=size_test)
        self.train, self.val = train_test_split(self.train, test_size=size_test)

        return self.train, self.test, self.val

    def convertToDataset(self, dataframe: pd.DataFrame, shuffle = True, batch_size = True):
        dataframe = dataframe.copy()
        labels = dataframe.pop('target')
        self.ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
        if shuffle:
            self.ds = self.ds.shuffle(buffer_size=len(dataframe))
        self.ds = self.ds.batch(batch_size)
        self.ds = self.ds.prefetch(batch_size)
        return self.ds

    def getNormalizationLayer(self, name, dataset):
        self.normalizer = preprocessing.Normalization()

        feature_df = dataset.map(lambda  x, y: x[name])
        self.normalizer.adapt(feature_df)
        return self.normalizer

    def getCategoryEncodingLayer(self, name, dataset, dtype, max_tokens=None):
        if dtype == 'string':
            index = preprocessing.StringLookup(max_tokens=max_tokens)
        else:
            index = preprocessing.IntegerLookup(max_tokens=max_tokens)
        feature_ds = dataset.map(lambda  x, y: x[name])
        index.adapt(feature_ds)
        encoder = preprocessing.CategoryEncoding(num_tokens=index.vocabulary_size())
        return lambda feature: encoder(index(feature))

    def getProcessedData(self):
        return self.samples, self.model_samples

class DarknetDataProcessing:
    def __init__(self, data, path):
        self.data = data
        self.path = path

    def doPreProcessing(self):
        # Correction of lables

        samples = self.data.copy()

        traffic_labels = samples['Label'].unique()
        traffic_type_labels = samples['Label.1'].unique()

        samples['Label.1'].loc[samples['Label.1'] == 'AUDIO-STREAMING'] = 'Audio-Streaming'
        samples['Label.1'].loc[samples['Label.1'] == 'File-transfer'] = 'File-Transfer'
        samples['Label.1'].loc[samples['Label.1'] == 'Video-streaming'] = 'Video-Streaming'

        traffic_type_labels = samples['Label.1'].unique()

        samples['Label'].loc[(samples['Label'] == 'Non-Tor') | (samples['Label'] == 'NonVPN')] = 'Benign'
        samples['Label'].loc[(samples['Label'] == 'Tor') | (samples['Label'] == 'VPN')] = 'Darknet'

        traffic_type_labels = samples['Label'].unique()

        hours = []
        for timestamp in samples['Timestamp']:
            hour = int(timestamp.split()[1].split(':')[0])
            hours.append(hour)
        samples['hour'] = hours

        ips_grams = {
            'src': {'one': [], 'two': [], 'three': []},
            'dst': {'one': [], 'two': [], 'three': []},
        }

        for src_ip, dst_ip in zip(samples['Src IP'], samples['Dst IP']):
            src_one, src_two, src_three = createGrams(src_ip)
            ips_grams['src']['one'].append(src_one)
            ips_grams['src']['two'].append(src_two)
            ips_grams['src']['three'].append(src_three)

            dst_one, dst_two, dst_three = createGrams(dst_ip)
            ips_grams['dst']['one'].append(dst_one)
            ips_grams['dst']['two'].append(dst_two)
            ips_grams['dst']['three'].append(dst_three)

        samples['src_ip_1gram'] = ips_grams['src']['one']
        samples['src_ip_2gram'] = ips_grams['src']['two']
        samples['src_ip_3gram'] = ips_grams['src']['three']

        samples['dst_ip_1gram'] = ips_grams['dst']['one']
        samples['dst_ip_2gram'] = ips_grams['dst']['two']
        samples['dst_ip_3gram'] = ips_grams['dst']['three']
        print(samples[["Src IP", "src_ip_1gram", "src_ip_2gram", "src_ip_3gram"]][200:205])
        print(samples[["Dst IP", "dst_ip_1gram", "dst_ip_2gram", "dst_ip_3gram"]][:5])

        ips = np.concatenate((samples['Src IP'].unique(), samples['Dst IP'].unique()))
        cat_ip_info = CatIPInformation("de30fe3213f197", ips)
        ips_dict = cat_ip_info.getIpsDict()

        ips_tuple = zip(samples['Src IP'], samples['Dst IP'])

        dst_ip_country = []
        src_ip_country = []
        src_bogon = []
        dst_bogon = []

        for src_ip, dst_ip in tqdm(ips_tuple, total=len(samples['Src IP'])):
            if 'country' in ips_dict[dst_ip].keys():
                dst_ip_country.append(ips_dict[dst_ip]['country'])
            else:
                dst_ip_country.append('')

            if 'country' in ips_dict[src_ip].keys():
                src_ip_country.append(ips_dict[src_ip]['country'])
            else:
                src_ip_country.append('')

            if 'bogon' in ips_dict[dst_ip].keys():
                dst_bogon.append(ips_dict[dst_ip]['bogon'])
            else:
                dst_bogon.append(False)

            if 'bogon' in ips_dict[src_ip].keys():
                src_bogon.append(ips_dict[src_ip]['bogon'])
            else:
                src_bogon.append(False)

        samples['dst_ip_country'] = dst_ip_country
        samples['src_ip_country'] = src_ip_country
        samples['dst_bogon'] = dst_bogon
        samples['src_bogon'] = src_bogon

        real_columns = ['Flow Duration', 'Total Fwd Packet', 'Total Bwd packets', 'Total Length of Fwd Packet',
                        'Total Length of Bwd Packet', 'Fwd Packet Length Max', 'Fwd Packet Length Min',
                        'Fwd Packet Length Mean', 'Fwd Packet Length Std', 'Bwd Packet Length Max',
                        'Bwd Packet Length Min', 'Bwd Packet Length Mean', 'Bwd Packet Length Std', 'Flow Bytes/s',
                        'Flow Packets/s', 'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min',
                        'Fwd IAT Total', 'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min',
                        'Bwd IAT Total',
                        'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min', 'Fwd PSH Flags',
                        'Bwd PSH Flags',
                        'Fwd URG Flags', 'Bwd URG Flags', 'Fwd Header Length', 'Bwd Header Length', 'Fwd Packets/s',
                        'Bwd Packets/s', 'Packet Length Min', 'Packet Length Max', 'Packet Length Mean',
                        'Packet Length Std', 'Packet Length Variance', 'FIN Flag Count', 'SYN Flag Count',
                        'RST Flag Count', 'PSH Flag Count', 'ACK Flag Count', 'URG Flag Count', 'CWE Flag Count',
                        'ECE Flag Count', 'Down/Up Ratio', 'Average Packet Size', 'Fwd Segment Size Avg',
                        'Bwd Segment Size Avg', 'Fwd Bytes/Bulk Avg', 'Fwd Packet/Bulk Avg', 'Fwd Bulk Rate Avg',
                        'Bwd Bytes/Bulk Avg', 'Bwd Packet/Bulk Avg', 'Bwd Bulk Rate Avg', 'Subflow Fwd Packets',
                        'Subflow Fwd Bytes', 'Subflow Bwd Packets', 'Subflow Bwd Bytes', 'FWD Init Win Bytes',
                        'Bwd Init Win Bytes', 'Fwd Act Data Pkts', 'Fwd Seg Size Min']
        is_na_cols = samples.columns[samples.isna().sum() > 0]
        print(samples.isna().sum()[is_na_cols])

        samples = samples.dropna()
        print(samples.isna().sum()[is_na_cols])

        samples[real_columns] = samples[real_columns].astype(np.float64)
        samples[real_columns] = samples[real_columns].replace([np.inf, -np.inf], np.nan)
        samples[real_columns] = samples[real_columns].dropna()

        model_samples = samples.copy()

        del model_samples['Flow ID']
        del model_samples['Timestamp']
        del model_samples['Src IP']
        del model_samples['Dst IP']

        cols = np.concatenate((model_samples.columns[81:], model_samples.columns[:81]))
        model_samples = model_samples[cols]

        hash_enc_cols = ['src_ip_1gram', 'src_ip_2gram', 'src_ip_3gram', 'dst_ip_1gram',
                         'dst_ip_2gram', 'dst_ip_3gram']
        ord_enc_cols = ['src_ip_country', 'dst_ip_country']

        print("[!] - Encoding Data. May take a while to process")
        hash_enc = HashingEncoder(cols=hash_enc_cols, n_components=100).fit(model_samples)
        model_samples = hash_enc.transform(model_samples)
        print(model_samples.head())

        ord_enc = OrdinalEncoder()
        ord_enc.fit(model_samples[ord_enc_cols])
        model_samples[ord_enc_cols] = ord_enc.transform(model_samples[ord_enc_cols])
        model_samples[ord_enc_cols] = model_samples[ord_enc_cols].astype(int)

        # scaler = StandardScaler().fit(model_samples[real_columns])
        # model_samples[real_columns] = scaler.transform(model_samples[real_columns])
        # print(model_samples[real_columns].head())

        model_samples['src_bogon'] = np.where(model_samples['src_bogon'], 1, 0)
        model_samples['dst_bogon'] = np.where(model_samples['dst_bogon'], 1, 0)



        self.samples = samples.dropna()
        self.model_samples = model_samples.dropna()



        self.model_samples.columns = self.model_samples.columns.str.replace(' ', '_')

        print(samples[samples.columns[samples.isna().sum() > 0]].isna().sum())

    def getProcessedData(self)-> pd.DataFrame:
        return self.samples, self.model_samples
