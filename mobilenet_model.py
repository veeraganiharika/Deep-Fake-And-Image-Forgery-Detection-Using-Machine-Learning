"""
MobileNet Feature Extractor for Image Forgery Detection
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import os


class MobileNetFeatureExtractor:
    """MobileNet-based feature extractor and classifier"""
    
    def __init__(self, input_shape=(224, 224, 3), num_classes=2):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        self.feature_extractor = None
        
    def build_model(self, trainable_layers=20):
        """Build MobileNet model with custom classification head"""
        
        # Load pre-trained MobileNet without top layers
        base_model = MobileNet(
            input_shape=self.input_shape,
            include_top=False,
            weights='imagenet'
        )
        
        # Freeze early layers, train later layers
        for layer in base_model.layers[:-trainable_layers]:
            layer.trainable = False
        for layer in base_model.layers[-trainable_layers:]:
            layer.trainable = True
        
        # Build custom classification head
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(256, activation='relu', name='fc1')(x)
        x = Dropout(0.5)(x)
        x = Dense(128, activation='relu', name='features')(x)  # Feature layer
        x = Dropout(0.3)(x)
        predictions = Dense(self.num_classes, activation='softmax', name='predictions')(x)
        
        # Create model
        self.model = Model(inputs=base_model.input, outputs=predictions)
        
        # Create feature extractor (up to 'features' layer)
        self.feature_extractor = Model(
            inputs=self.model.input,
            outputs=self.model.get_layer('features').output
        )
        
        return self.model
    
    def compile_model(self, learning_rate=0.001):
        """Compile the model"""
        
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("Model compiled successfully!")
        
    def train(self, X_train, y_train, X_val=None, y_val=None, 
              epochs=50, batch_size=32, save_path='models/mobilenet_best.h5'):
        """Train the model"""
        
        # Create save directory
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Callbacks
        callbacks = [
            ModelCheckpoint(
                save_path,
                monitor='val_accuracy' if X_val is not None else 'accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss' if X_val is not None else 'loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # Prepare validation data
        validation_data = (X_val, y_val) if X_val is not None else None
        
        # Train
        history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X)
    
    def extract_features(self, X):
        """Extract 128-dimensional features from images"""
        if self.feature_extractor is None:
            # If model was loaded, recreate feature extractor
            self.feature_extractor = Model(
                inputs=self.model.input,
                outputs=self.model.get_layer('features').output
            )
        return self.feature_extractor.predict(X)
    
    def evaluate(self, X_test, y_test):
        """Evaluate the model"""
        return self.model.evaluate(X_test, y_test, verbose=1)
    
    def save_model(self, filepath):
        """Save the model"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a saved model"""
        self.model = load_model(filepath)
        
        # Recreate feature extractor
        self.feature_extractor = Model(
            inputs=self.model.input,
            outputs=self.model.get_layer('features').output
        )
        
        print(f"Model loaded from {filepath}")
    
    def get_model_summary(self):
        """Print model summary"""
        if self.model:
            self.model.summary()
        else:
            print("Model not built yet!")


if __name__ == "__main__":
    # Test the model
    print("Testing MobileNet Feature Extractor...")
    
    extractor = MobileNetFeatureExtractor()
    model = extractor.build_model(trainable_layers=20)
    
    print("\nModel Summary:")
    extractor.get_model_summary()
    
    print("\n✓ MobileNet model created successfully!")
