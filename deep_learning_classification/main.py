from src.dataset import Darknet
from src.models import ClassificationModel
from src.utils.functions import mkdir_if_not_exists, verify_existence_data
import tensorflow as tf
from tqdm import tqdm

def main():
    data_path = f'dataset/Darknet/'
    #TODO {colocar para pegar o dataset por parâmetro}
    darknet_dataset = Darknet(data_path)
    if not verify_existence_data(f"{data_path}/processed/darknet_dataset_processed_encoded.csv") and not verify_existence_data(f"{data_path}/processed/darknet_dataset_processed.csv"):
        darknet_dataset.exportProcessedData()
    darknet_data, darknet_data_models = darknet_dataset.getProcessedData()



    train, test, val = darknet_dataset.getTrainTestValData(0.33, False)


    train_ds = darknet_dataset.convertToDataset(train, batch_size=256)
    test_ds = darknet_dataset.convertToDataset(test, shuffle=False, batch_size=256)
    val_ds = darknet_dataset.convertToDataset(val, shuffle=False, batch_size=256)

    data_columns = darknet_data_models.columns
    integer_category_columns = data_columns[:108]
    real_columns = data_columns[108:len(data_columns) - 2]

    inputs = []
    encoded_features =[]




    print("[!] - Normalizating real columns", end="\n")
    for i in tqdm(range(len(real_columns))):
        numeric_col = tf.keras.Input(shape=(1, ), name=real_columns[i])
        normalization_layer = darknet_dataset.getNormalizationLayer(real_columns[i], train_ds)
        encoded_numeric_col = normalization_layer(numeric_col)
        inputs.append(numeric_col)
        encoded_features.append(encoded_numeric_col)

    print("[!] - Encoding integer and string layers", end="\n")
    for i in tqdm(range(len(integer_category_columns))):
        categorical_col = tf.keras.Input(shape=(1, ), name=integer_category_columns[i], dtype='int64')
        encoding_layer = darknet_dataset.getCategoryEncodingLayer(integer_category_columns[i], train_ds, dtype='int64', max_tokens=5)
        encoded_categorical_col = encoding_layer(categorical_col)
        inputs.append(categorical_col)
        encoded_features.append(encoded_categorical_col)

    model_classification = ClassificationModel(inputs, encoded_features, 'relu', "characterization")
    mkdir_if_not_exists('imgs/')

    model_classification.fitModel(train_ds, val_ds)
    model_classification.evaluate(test_ds)

    tf.keras.utils.plot_model(model_classification.getModel(), show_shapes=True, rankdir="LR", to_file="imgs/model.png")


if __name__ == '__main__':
    main()


