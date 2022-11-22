from os import getcwd
from os.path import join as path_join

import numpy as np
from tensorflow import float32, py_function, Tensor
from tensorflow.data import Dataset
from tensorflow.keras import Model, Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.layers import (
    BatchNormalization,
    Convolution1D,
    Concatenate,
    Dense,
    Dropout,
    Flatten,
    Input,
    InputLayer,
    LeakyReLU,
    MaxPooling1D,
    Normalization,
    Softmax
)
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import L2
from wfdb import rdsamp

from utils.get_class_weights import get_class_weights
from utils.get_paths_and_my_labels_splitted import get_paths_and_labels


BATCH_NORMALIZATION_AXIS = -1
CONVOLUTIONAL_LAYERS_KWARGS = {
    'padding': 'same',
    'activation': 'linear',
}
DROPOUT_RATE = 0.5
FILTERED_DATA_FLAG = False
FLAG_INCEPTION_MODEL = False
FULLY_CONNECTED_LAYERS_KWARGS = {
    'activation': 'linear'
}
INPUTS_STANDARDIZATION_LAYER_NAME = 'inputs_standardization'
LEAKY_RELU_NEGATIVE_SLOPE = 0.01
MINI_BATCH_SIZE = 256
MODEL_FILE_PATH = path_join(getcwd(), 'trained_model.h5')
N_CHANNELS = 12
N_CLASSES = 5
N_EPOCHS = 100
N_SAMPLES_PER_SIGNAl = 1000
L2_REGULARIZATION_LAMBDA = 0.01
LEARNING_RATE = 1e-3
REGULARIZATION_KWARGS = {
    'kernel_regularizer': L2(l2=L2_REGULARIZATION_LAMBDA),
    'bias_regularizer': L2(l2=L2_REGULARIZATION_LAMBDA)
}
TRAINING_SET_FILE_PATHS, TRAINING_SET_LABELS, VALIDATION_SET_FILE_PATHS, \
    VALIDATION_SET_LABELS, TEST_SET_FILE_PATHS, TEST_SET_LABELS \
        = get_paths_and_labels(FILTERED_DATA_FLAG)


def build_convolutional_block(n_filters: int) -> Model:
    """
    Build an instance of a convolutional layer with a leaky-ReLU activation
    function batch-normalization in-between convolution and activation. 
    """
    return Sequential(
        layers=[
            Convolution1D(
                filters=n_filters,
                kernel_size=3,
                **CONVOLUTIONAL_LAYERS_KWARGS,
                **REGULARIZATION_KWARGS
            ),
            BatchNormalization(axis=BATCH_NORMALIZATION_AXIS),
            LeakyReLU(alpha=LEAKY_RELU_NEGATIVE_SLOPE),

            MaxPooling1D(),
        ]
    )


def build_fully_connected_block(n_output_neurons: int) -> Model:
    """
    Build an instance of a fully-connected layer with LeakyReLU and Dropout.
    """
    return Sequential(
        layers=[
            Dense(
                units=n_output_neurons,
                **FULLY_CONNECTED_LAYERS_KWARGS,
                **REGULARIZATION_KWARGS
            ),
            LeakyReLU(alpha=LEAKY_RELU_NEGATIVE_SLOPE),
            Dropout(rate=DROPOUT_RATE),
        ]
    )


def propagate_through_convolutional_block(n_filters: int, input: Tensor) -> Model:
    """
    Get the output of the convolutional block in Inception model
    """
    layer_1_out = Convolution1D(
                filters=n_filters,
                kernel_size=1,
                **CONVOLUTIONAL_LAYERS_KWARGS,
                **REGULARIZATION_KWARGS
            )(input)
    layer_1_out = BatchNormalization(
                axis=BATCH_NORMALIZATION_AXIS
            )(layer_1_out)
    layer_1_out = LeakyReLU(
                alpha=LEAKY_RELU_NEGATIVE_SLOPE
            )(layer_1_out)
    layer_2_out = Convolution1D(
                filters=n_filters,
                kernel_size=3,
                **CONVOLUTIONAL_LAYERS_KWARGS,
                **REGULARIZATION_KWARGS
            )(input)
    layer_2_out = BatchNormalization(
                axis=BATCH_NORMALIZATION_AXIS
            )(layer_2_out)
    layer_2_out = LeakyReLU(
                alpha=LEAKY_RELU_NEGATIVE_SLOPE
            )(layer_2_out)
    layer_3_out = Convolution1D(
                filters=n_filters,
                kernel_size=5,
                **CONVOLUTIONAL_LAYERS_KWARGS,
                **REGULARIZATION_KWARGS
            )(input)
    layer_3_out = BatchNormalization(
                axis=BATCH_NORMALIZATION_AXIS
            )(layer_3_out)
    layer_3_out = LeakyReLU(
                alpha=LEAKY_RELU_NEGATIVE_SLOPE
            )(layer_3_out)
    layer_4_out = MaxPooling1D(
                pool_size=3,
                strides=1,
                padding='same'
            )(input)
    output = Concatenate()(
                [layer_1_out, layer_2_out, layer_3_out, layer_4_out]
            )
    output = MaxPooling1D()(output)
    return output


def propagate_through_fully_connected_block(n_output_neurons: int, input: Tensor) -> Model:
    """
    Get the output of the fully-connected block in Inception model
    """
    return Sequential(
        layers=[
            Dense(
                units=n_output_neurons,
                **FULLY_CONNECTED_LAYERS_KWARGS,
                **REGULARIZATION_KWARGS
            ),
            LeakyReLU(alpha=LEAKY_RELU_NEGATIVE_SLOPE),
            Dropout(rate=DROPOUT_RATE),
        ]
    )(input)


def build_model_architecture(flag_inception: bool) -> Model:
    """
    Instantiate a model with the defined architecture.
    """
    if flag_inception:
        in_tensor = Input(shape=(N_SAMPLES_PER_SIGNAl, N_CHANNELS))
        out_tensor = propagate_through_convolutional_block(n_filters=16, input=in_tensor)
        out_tensor = propagate_through_convolutional_block(n_filters=32, input=out_tensor)
        out_tensor = propagate_through_convolutional_block(n_filters=64, input=out_tensor)
        out_tensor = propagate_through_convolutional_block(n_filters=64, input=out_tensor)
        out_tensor = propagate_through_convolutional_block(n_filters=64, input=out_tensor)
        out_tensor = Flatten()(out_tensor)
        out_tensor = propagate_through_fully_connected_block(n_output_neurons=8, input=out_tensor)
        # out_tensor = propagate_through_fully_connected_block(n_output_neurons=32, input=out_tensor)
        out_tensor = Dense(
            units=N_CLASSES,
            activation='linear',
            **REGULARIZATION_KWARGS
        )(out_tensor)
        out_tensor = Softmax()(out_tensor)
        return Model(in_tensor, out_tensor)
    else:
        return Sequential(
            layers=[
                InputLayer(input_shape=(N_SAMPLES_PER_SIGNAl, N_CHANNELS)),
                # Normalization(axis=-1, name=INPUTS_STANDARDIZATION_LAYER_NAME),
                build_convolutional_block(n_filters=16), 
                build_convolutional_block(n_filters=32),
                build_convolutional_block(n_filters=64),
                build_convolutional_block(n_filters=128),
                build_convolutional_block(n_filters=256),
                Flatten(),
                build_fully_connected_block(n_output_neurons=256),  
                build_fully_connected_block(n_output_neurons=64), 
                Dense(
                    units=N_CLASSES,
                    activation='linear',
                    **REGULARIZATION_KWARGS
                ),
                Softmax()
            ]
        )


def load_ecg_leads_from_file(path: Tensor) -> Tensor:
    """
    Load signals representing ECG leads from the file whose path is given as
    input.
    """
    if FILTERED_DATA_FLAG:
        return np.load(file=bytes.decode(path.numpy()))
    else:
        return rdsamp(record_name=bytes.decode(path.numpy()))[0]
    


def turn_paths_dataset_into_batched_labelled_signals_dataset(
        file_paths_dataset: Dataset,
        labels_dataset: Dataset
) -> Dataset:
    """
    Turn the input file paths dataset into the dataset of corresponding signals.
    """
    return Dataset.zip(
        datasets=(
            (
                file_paths_dataset
                .map(
                    map_func=lambda file_path: py_function(
                        func=load_ecg_leads_from_file,
                        inp=[file_path, ],
                        Tout=float32
                    )
                )
                .batch(batch_size=MINI_BATCH_SIZE)
            ),
            (
                labels_dataset
                .batch(batch_size=MINI_BATCH_SIZE)
            )
            # TODO weights for each sample
        )
    )


def main() -> None:
    """
    Do everything
    """
    training_file_paths_dataset = Dataset.from_tensor_slices(tensors=TRAINING_SET_FILE_PATHS)
    training_labels_dataset = Dataset.from_tensor_slices(tensors=TRAINING_SET_LABELS)
    validation_file_paths_dataset = Dataset.from_tensor_slices(tensors=VALIDATION_SET_FILE_PATHS)
    validation_labels_dataset = Dataset.from_tensor_slices(tensors=VALIDATION_SET_LABELS)
    test_file_paths_dataset = Dataset.from_tensor_slices(tensors=TEST_SET_FILE_PATHS)
    test_labels_dataset = Dataset.from_tensor_slices(tensors=TEST_SET_LABELS)

    training_set = turn_paths_dataset_into_batched_labelled_signals_dataset(
        file_paths_dataset=training_file_paths_dataset,
        labels_dataset=training_labels_dataset
    )
    validation_set = turn_paths_dataset_into_batched_labelled_signals_dataset(
        file_paths_dataset=validation_file_paths_dataset,
        labels_dataset=validation_labels_dataset
    )
    test_set = turn_paths_dataset_into_batched_labelled_signals_dataset(
        file_paths_dataset=test_file_paths_dataset,
        labels_dataset=test_labels_dataset
    )

    # for mini_batch_data in training_set:
    #     print(
    #         'mini batch with samples with shape',
    #         mini_batch_data[0].shape,
    #         'and labels with shape',
    #         mini_batch_data[1].shape
    #     )
    #     break

    model = build_model_architecture(flag_inception=FLAG_INCEPTION_MODEL)
    model.summary()

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss=SparseCategoricalCrossentropy(),
        metrics=[SparseCategoricalAccuracy()]
    )

    # # computing the initial normalization layer's statistics (the
    # # standardization mean and variance of each input feature) over the
    # # training set:
    # model.get_layer(
    #     name=INPUTS_STANDARDIZATION_LAYER_NAME
    # ).adapt(
    #     data=training_set.map(
    #         # selecting only samples:
    #         lambda *mini_batch_data: mini_batch_data[0]
    #     )
    # )

    model_checkpoint_callback = ModelCheckpoint(
        filepath=MODEL_FILE_PATH,
        save_best_only=True,
        monitor='val_sparse_categorical_accuracy',
        mode='max',
    )

    model.fit(
        x=training_set,
        epochs=N_EPOCHS,
        callbacks=[
            TensorBoard(log_dir='tensorboard_runs'), 
            model_checkpoint_callback
        ],
        validation_data=validation_set,
        #class_weight=get_class_weights()
    )

    # TODO: tune hyperparameters to optimize performance on the validation
    # set

    model.evaluate(
        x=test_set
    )

    # model.save(filepath=MODEL_FILE_PATH, overwrite=False)

    for mini_batch_index, mini_batch_data in enumerate(test_set):
        mini_batch_samples, mini_batch_labels = mini_batch_data
        mini_batch_predictions = model.predict_on_batch(
            x=mini_batch_samples
        )
        # TODO: plot predictions (mini_batch_predictions) vs expectations
        # (mini_batch_labels) over signals (test_set_mini_batch)


if __name__ == '__main__':
    main()