import tensorflow as tf
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Conv2D, Add, ReLU, BatchNormalization, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import DenseNet121

class BaseModel:
    def __init__(self, input_shape=(224, 224, 3), num_classes=2, learning_rate=0.001):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.model = None

    def build(self):
        """Override this method in subclasses to build different architectures."""
        raise NotImplementedError("Build method not implemented!")

    def compile(self):
        if self.model is None:
            raise ValueError("Model is not built yet. Call build() first.")
        self.model.compile(
            loss='binary_crossentropy',
            optimizer=Adam(learning_rate=self.learning_rate),
            metrics=['accuracy']
        )
        return self.model


class CNNModel(BaseModel):
    def build(self):
        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(self.num_classes, activation='softmax'))
        self.model = model
        return self.model


class PatchClassifier(BaseModel):
    def __init__(self, input_shape=(512, 512, 3), num_patch_classes=2, learning_rate=0.001):
        super().__init__(input_shape=input_shape, num_classes=num_patch_classes, learning_rate=learning_rate)

    def build(self):
        """
        Builds the DenseNet-121 based patch classifier model.
        """

        base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=self.input_shape)

        base_model.trainable = False

        x = base_model.output

        x = GlobalAveragePooling2D()(x)

        output = Dense(self.num_classes, activation='softmax')(x)

        self.model = Model(inputs=base_model.input, outputs=output, name="PatchClassifier")
        return self.model
