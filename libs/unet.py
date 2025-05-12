import tensorflow as tf
import segmentation_models as sm
from tensorflow.keras.layers import Input, Conv2D
from tensorflow.keras.models import Model

class SimpleUNet:
    def __init__(self, input_shape=(224, 224, 8), num_classes=1):
        self.input_shape = input_shape
        self.num_classes = num_classes

    def build_model(self):
        inputs = tf.keras.Input(shape=self.input_shape)

        # Downsample
        c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
        c1 = tf.keras.layers.Dropout(0.1)(c1)
        c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
        p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

        # Bottleneck
        c5 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
        c5 = tf.keras.layers.Dropout(0.2)(c5)
        c5 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

        # Upsample
        u6 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c5)
        u6 = tf.keras.layers.concatenate([u6, c1])
        c6 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
        c6 = tf.keras.layers.Dropout(0.1)(c6)
        c6 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

        # Output layer
        outputs = tf.keras.layers.Conv2D(self.num_classes, (1, 1), activation='sigmoid')(c6)

        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        return model

import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate

class DeeperUnet:
    def __init__(self, input_shape=(224, 224, 8), num_classes=1):
        self.input_shape = input_shape
        self.num_classes = num_classes

    def build_model(self):
        inputs = tf.keras.Input(shape=self.input_shape)
        
        # Downsampling path
        c0 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
        p0 = tf.keras.layers.MaxPooling2D((2, 2))(c0)
        
        c1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(p0)
        p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)
        
        c2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(p1)
        p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)
        
        c3 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(p2)
        p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)
        
        # Bottleneck
        c4 =tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(p3)
        
        # Upsampling path
        u1 = tf.keras.layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), activation='relu', padding='same', kernel_initializer='he_normal')(c4)
        u1 = tf.keras.layers.concatenate([u1, c3])
        
        u2 =tf.keras.layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), activation='relu', padding='same', kernel_initializer='he_normal')(u1)
        u2 = tf.keras.layers.concatenate([u2, c2])
        
        u3 = tf.keras.layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), activation='relu', padding='same', kernel_initializer='he_normal')(u2)
        u3 = tf.keras.layers.concatenate([u3, c1])

        u4 = tf.keras.layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), activation='relu', padding='same', kernel_initializer='he_normal')(u3)
        u4 = tf.keras.layers.concatenate([u4, c0])
        
        # Output layer
        outputs = tf.keras.layers.Conv2D(self.num_classes, (1, 1), activation='sigmoid', padding='same')(u4)
        
        # Create the model
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        return model

class DeeperUnet_dropout:
    def __init__(self, input_shape=(224, 224, 8), num_classes=1, dropout_rate=0.5):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate  # Add a parameter for dropout rate

    def build_model(self):
        inputs = tf.keras.Input(shape=self.input_shape)
        
        # Downsampling path
        c0 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
        c0 = tf.keras.layers.Dropout(self.dropout_rate)(c0)  # Dropout added here
        p0 = tf.keras.layers.MaxPooling2D((2, 2))(c0)
        
        c1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(p0)
        c1 = tf.keras.layers.Dropout(self.dropout_rate)(c1)  # Dropout added here
        p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)
        
        c2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(p1)
        c2 = tf.keras.layers.Dropout(self.dropout_rate)(c2)  # Dropout added here
        p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)
        
        c3 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(p2)
        c3 = tf.keras.layers.Dropout(self.dropout_rate)(c3)  # Dropout added here
        p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)
        
        # Bottleneck
        c4 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(p3)
        c4 = tf.keras.layers.Dropout(self.dropout_rate)(c4)  # Dropout added here
        
        # Upsampling path
        u1 = tf.keras.layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), activation='relu', padding='same', kernel_initializer='he_normal')(c4)
        u1 = tf.keras.layers.concatenate([u1, c3])
        u1 = tf.keras.layers.Dropout(self.dropout_rate)(u1)  # Dropout added here
        
        u2 = tf.keras.layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), activation='relu', padding='same', kernel_initializer='he_normal')(u1)
        u2 = tf.keras.layers.concatenate([u2, c2])
        u2 = tf.keras.layers.Dropout(self.dropout_rate)(u2)  # Dropout added here
        
        u3 = tf.keras.layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), activation='relu', padding='same', kernel_initializer='he_normal')(u2)
        u3 = tf.keras.layers.concatenate([u3, c1])
        u3 = tf.keras.layers.Dropout(self.dropout_rate)(u3)  # Dropout added here

        u4 = tf.keras.layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), activation='relu', padding='same', kernel_initializer='he_normal')(u3)
        u4 = tf.keras.layers.concatenate([u4, c0])
        u4 = tf.keras.layers.Dropout(self.dropout_rate)(u4)  # Dropout added here
        
        # Output layer
        outputs = tf.keras.layers.Conv2D(self.num_classes, (1, 1), activation='sigmoid', padding='same')(u4)
        
        # Create the model
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        return model

class SimpleAttentionUNet:
    def __init__(self, input_shape=(224, 224, 8), num_classes=1):
        self.input_shape = input_shape
        self.num_classes = num_classes

    def attention_gate(self, g, x, filters):
        # g is the gating signal, x is the skip connection
        theta_x = tf.keras.layers.Conv2D(filters, (1, 1), strides=(1, 1), padding='same')(x)
        phi_g = tf.keras.layers.Conv2D(filters, (1, 1), strides=(1, 1), padding='same')(g)
        add_xg = tf.keras.layers.add([theta_x, phi_g])
        relu_xg = tf.keras.layers.Activation('relu')(add_xg)
        psi = tf.keras.layers.Conv2D(1, (1, 1), padding='same')(relu_xg)
        sigmoid_xg = tf.keras.layers.Activation('sigmoid')(psi)
        attention = tf.keras.layers.multiply([x, sigmoid_xg])
        return attention

    def build_model(self):
        inputs = tf.keras.Input(shape=self.input_shape)

        # Downsample
        c1 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
        c1 = tf.keras.layers.Dropout(0.1)(c1)
        c1 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
        p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

        c2 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
        c2 = tf.keras.layers.Dropout(0.1)(c2)
        c2 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
        p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)

        c3 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
        c3 = tf.keras.layers.Dropout(0.2)(c3)
        c3 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
        p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)

        c4 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
        c4 = tf.keras.layers.Dropout(0.2)(c4)
        c4 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
        p4 = tf.keras.layers.MaxPooling2D((2, 2))(c4)

        # Bottleneck
        c5 = tf.keras.layers.Conv2D(1024, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
        c5 = tf.keras.layers.Dropout(0.3)(c5)
        c5 = tf.keras.layers.Conv2D(1024, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

        # Upsample
        u6 = tf.keras.layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(c5)
        att6 = self.attention_gate(u6, c4, 512)
        u6 = tf.keras.layers.concatenate([u6, att6])
        c6 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
        c6 = tf.keras.layers.Dropout(0.2)(c6)
        c6 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

        u7 = tf.keras.layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c6)
        att7 = self.attention_gate(u7, c3, 256)
        u7 = tf.keras.layers.concatenate([u7, att7])
        c7 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
        c7 = tf.keras.layers.Dropout(0.2)(c7)
        c7 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

        u8 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c7)
        att8 = self.attention_gate(u8, c2, 128)
        u8 = tf.keras.layers.concatenate([u8, att8])
        c8 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
        c8 = tf.keras.layers.Dropout(0.1)(c8)
        c8 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

        u9 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c8)
        att9 = self.attention_gate(u9, c1, 64)
        u9 = tf.keras.layers.concatenate([u9, att9])
        c9 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
        c9 = tf.keras.layers.Dropout(0.1)(c9)
        c9 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

        # Output layer
        outputs = tf.keras.layers.Conv2D(self.num_classes, (1, 1), activation='sigmoid')(c9)

        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        return model

class ResNet50Unet:
    def __init__(self, input_shape=(224, 224, 8), classes=1, activation='sigmoid', encoder_weights='imagenet'):     
        self.input_shape = input_shape
        self.classes = classes
        self.activation = activation
        self.encoder_weights = encoder_weights
        self.num_input_channels = input_shape[-1]

    def build_model(self):
        actual_input = Input(shape=self.input_shape, name="multi_channel_input")
        if self.num_input_channels == 3 and self.encoder_weights == 'imagenet':
            # If input is already 3 channels, and we want to use imagenet weights,
            # we can directly use this input for the base_model.
            # However, the user's example explicitly uses the Conv2D layer,
            # so we will follow that pattern but acknowledge this edge case.
            print("Input channels are 3; the 1x1 Conv2D still maps 3->3 as per example structure.")
            mapped_input = Conv2D(3, (1, 1), padding='same', name='input_channel_mapper', activation='relu')(actual_input)
        elif self.num_input_channels > 0 : # General case for N channels
            mapped_input = Conv2D(3, (1, 1), padding='same', name='input_channel_mapper', activation='relu')(actual_input)
        else:
            raise ValueError("Number of input channels must be positive.")
        
        # Create the U-Net base model.
        # This base model will internally expect a 3-channel input because its
        # encoder (ResNet50 with 'imagenet' weights) is built for 3 channels.
        # We don't pass input_shape to it if we are calling it on a tensor.
        base_unet_model = sm.Unet(backbone_name='resnet50',
                                    encoder_weights=self.encoder_weights,
                                    # input_shape=(self.input_shape[0], self.input_shape[1], 3), # Not needed if called on a tensor
                                    classes=self.classes,
                                    activation=self.activation)

        # Pass the 3-channel mapped input to the U-Net base model
        segmentation_output = base_unet_model(mapped_input)

        # Create the final model that links the N-channel input to the segmentation output
        model = Model(inputs=actual_input, outputs=segmentation_output, name=f"ResNet50_Unet_{self.num_input_channels}Ch_Input")

        return model