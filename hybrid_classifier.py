"""
Hybrid Classifier for Image Forgery Detection
Combines multiple ML algorithms with CNN features
Fixed for scikit-learn 1.2+ compatibility
"""

import numpy as np
import joblib
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    VotingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')


class HybridClassifier:
    """Hybrid classifier combining multiple ML algorithms"""
    
    def __init__(self, use_scaler=True):
        self.use_scaler = use_scaler
        self.scaler = StandardScaler() if use_scaler else None
        self.classifiers = {}
        self.voting_classifier = None
        self.initialize_classifiers()
    
    def initialize_classifiers(self):
        """Initialize all ML classifiers with optimized parameters"""
        
        print("🔧 Initializing ML classifiers...")
        
        # Base estimator for AdaBoost (required in newer versions)
        base_estimator = DecisionTreeClassifier(max_depth=3, random_state=42)
        
        self.classifiers = {
            'Random Forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                n_jobs=-1,
                random_state=42,
                class_weight='balanced'
            ),
            
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=5,
                min_samples_split=5,
                min_samples_leaf=2,
                subsample=0.8,
                random_state=42
            ),
            
            'AdaBoost': AdaBoostClassifier(
                estimator=base_estimator,
                n_estimators=100,
                learning_rate=0.8,
                random_state=42
            ),
            
            'Logistic Regression': LogisticRegression(
                C=1.0,
                max_iter=1000,
                solver='lbfgs',
                n_jobs=-1,
                random_state=42,
                class_weight='balanced'
            ),
            
            'SVM': SVC(
                C=1.0,
                kernel='rbf',
                gamma='scale',
                probability=True,
                random_state=42,
                class_weight='balanced'
            ),
            
            'KNN': KNeighborsClassifier(
                n_neighbors=7,
                weights='distance',
                metric='minkowski',
                n_jobs=-1
            ),
            
            'Naive Bayes': GaussianNB(
                var_smoothing=1e-9
            ),
            
            'Decision Tree': DecisionTreeClassifier(
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                random_state=42,
                class_weight='balanced'
            ),
            
            'MLP': MLPClassifier(
                hidden_layer_sizes=(256, 128, 64),
                activation='relu',
                solver='adam',
                alpha=0.0001,
                batch_size=32,
                learning_rate='adaptive',
                learning_rate_init=0.001,
                max_iter=500,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=20,
                random_state=42
            ),
            
            'XGBoost': None
        }
        
        # Try to import and add XGBoost
        try:
            from xgboost import XGBClassifier
            self.classifiers['XGBoost'] = XGBClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=6,
                min_child_weight=1,
                subsample=0.8,
                colsample_bytree=0.8,
                gamma=0,
                reg_alpha=0.1,
                reg_lambda=1,
                n_jobs=-1,
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss'
            )
            print("  ✓ XGBoost initialized")
        except ImportError:
            print("  ⚠ XGBoost not available, skipping...")
            del self.classifiers['XGBoost']
        
        # Remove None classifiers
        self.classifiers = {k: v for k, v in self.classifiers.items() if v is not None}
        
        print(f"  ✓ Initialized {len(self.classifiers)} classifiers")
    
    def get_all_predictions(self, features):
        """
        Get predictions from all individual classifiers
        
        Args:
            features: Feature array (can be single sample or batch)
            
        Returns:
            Dictionary with predictions from each classifier
        """
        # Ensure features is 2D
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
        
        # Scale features if scaler is available
        if self.scaler is not None:
            features_scaled = self.scaler.transform(features)
        else:
            features_scaled = features
        
        predictions = {}
        
        # Get predictions from each classifier
        for name, clf in self.classifiers.items():
            if clf is None:
                continue
                
            try:
                # Get prediction and probability
                pred = clf.predict(features_scaled)
                prob = clf.predict_proba(features_scaled)
                
                predictions[name] = {
                    'prediction': int(pred[0]),
                    'probability': prob[0].tolist()
                }
            except Exception as e:
                predictions[name] = {
                    'error': str(e)
                }
        
        # Get voting classifier prediction if available
        if self.voting_classifier is not None:
            try:
                pred = self.voting_classifier.predict(features_scaled)
                prob = self.voting_classifier.predict_proba(features_scaled)
                
                predictions['Voting Classifier'] = {
                    'prediction': int(pred[0]),
                    'probability': prob[0].tolist()
                }
            except Exception as e:
                predictions['Voting Classifier'] = {
                    'error': str(e)
                }
        
        return predictions
    
    def train(self, X_train, y_train, X_test=None, y_test=None):
        """Train all classifiers"""
        
        print(f"\n{'='*80}")
        print("📚 TRAINING ML CLASSIFIERS")
        print(f"{'='*80}")
        print(f"Training samples: {len(X_train)}")
        if X_test is not None:
            print(f"Test samples: {len(X_test)}")
        print(f"{'='*80}\n")
        
        # Scale features if enabled
        if self.use_scaler:
            print("🔄 Scaling features...")
            X_train = self.scaler.fit_transform(X_train)
            if X_test is not None:
                X_test = self.scaler.transform(X_test)
            print("  ✓ Features scaled\n")
        
        # Train each classifier
        results = {}
        
        for i, (name, clf) in enumerate(self.classifiers.items(), 1):
            print(f"[{i}/{len(self.classifiers)}] Training {name}...")
            
            try:
                # Train
                clf.fit(X_train, y_train)
                
                # Evaluate on training set
                train_pred = clf.predict(X_train)
                train_acc = accuracy_score(y_train, train_pred)
                
                # Evaluate on test set if provided
                test_acc = None
                if X_test is not None and y_test is not None:
                    test_pred = clf.predict(X_test)
                    test_acc = accuracy_score(y_test, test_pred)
                
                results[name] = {
                    'train_accuracy': train_acc,
                    'test_accuracy': test_acc,
                    'classifier': clf
                }
                
                print(f"  ✓ Training Accuracy: {train_acc:.4f}", end="")
                if test_acc is not None:
                    print(f" | Test Accuracy: {test_acc:.4f}")
                else:
                    print()
                
            except Exception as e:
                print(f"  ✗ Error training {name}: {str(e)}")
                results[name] = {
                    'train_accuracy': 0,
                    'test_accuracy': 0,
                    'classifier': None,
                    'error': str(e)
                }
        
        print(f"\n{'='*80}")
        print("📊 TRAINING SUMMARY")
        print(f"{'='*80}")
        
        # Sort by test accuracy if available, else by train accuracy
        if X_test is not None:
            sorted_results = sorted(
                results.items(),
                key=lambda x: x[1]['test_accuracy'] if x[1]['test_accuracy'] else 0,
                reverse=True
            )
        else:
            sorted_results = sorted(
                results.items(),
                key=lambda x: x[1]['train_accuracy'],
                reverse=True
            )
        
        for rank, (name, result) in enumerate(sorted_results, 1):
            if result['classifier'] is not None:
                print(f"{rank}. {name:20s} - Train: {result['train_accuracy']:.4f}", end="")
                if result['test_accuracy'] is not None:
                    print(f" | Test: {result['test_accuracy']:.4f}")
                else:
                    print()
        
        print(f"{'='*80}\n")
        
        return results
    
    def create_voting_classifier(self, top_n=5):
        """Create ensemble voting classifier from top N performers"""
        
        print(f"\n🗳️ Creating Voting Classifier (Top {top_n} models)...")
        
        # Get top N classifiers
        estimators = list(self.classifiers.items())[:top_n]
        
        self.voting_classifier = VotingClassifier(
            estimators=estimators,
            voting='soft',
            n_jobs=-1
        )
        
        print(f"  ✓ Voting classifier created with: {[name for name, _ in estimators]}")
        
        return self.voting_classifier
    
    def save_models(self, save_dir='models'):
        """Save all trained models"""
        
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"\n💾 Saving models to {save_dir}/...")
        
        # Save scaler
        if self.scaler is not None:
            scaler_path = os.path.join(save_dir, 'scaler.pkl')
            joblib.dump(self.scaler, scaler_path)
            print(f"  ✓ Scaler saved to {scaler_path}")
        
        # Save individual classifiers
        for name, clf in self.classifiers.items():
            if clf is not None:
                filename = name.lower().replace(' ', '_') + '.pkl'
                filepath = os.path.join(save_dir, filename)
                joblib.dump(clf, filepath)
                print(f"  ✓ {name} saved to {filepath}")
        
        # Save voting classifier if created
        if self.voting_classifier is not None:
            voting_path = os.path.join(save_dir, 'voting_classifier.pkl')
            joblib.dump(self.voting_classifier, voting_path)
            print(f"  ✓ Voting Classifier saved to {voting_path}")
        
        print(f"  ✓ All models saved successfully!\n")
    
    def load_models(self, save_dir='models'):
        """Load all trained models"""
        
        import os
        
        print(f"\n📂 Loading models from {save_dir}/...")
        
        # Load scaler
        scaler_path = os.path.join(save_dir, 'scaler.pkl')
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
            print(f"  ✓ Scaler loaded")
        
        # Load individual classifiers
        for name in list(self.classifiers.keys()):
            filename = name.lower().replace(' ', '_') + '.pkl'
            filepath = os.path.join(save_dir, filename)
            if os.path.exists(filepath):
                self.classifiers[name] = joblib.load(filepath)
                print(f"  ✓ {name} loaded")
        
        # Load voting classifier
        voting_path = os.path.join(save_dir, 'voting_classifier.pkl')
        if os.path.exists(voting_path):
            self.voting_classifier = joblib.load(voting_path)
            print(f"  ✓ Voting Classifier loaded")
        
        print(f"  ✓ All models loaded successfully!\n")
    
    def predict(self, X, classifier_name=None):
        """Make predictions using a specific classifier or voting classifier"""
        
        # Scale features if scaler is available
        if self.scaler is not None:
            X = self.scaler.transform(X)
        
        if classifier_name:
            # Use specific classifier
            if classifier_name in self.classifiers:
                return self.classifiers[classifier_name].predict(X)
            else:
                raise ValueError(f"Classifier '{classifier_name}' not found")
        else:
            # Use voting classifier
            if self.voting_classifier is not None:
                return self.voting_classifier.predict(X)
            else:
                raise ValueError("Voting classifier not initialized. Call create_voting_classifier() first.")
    
    def evaluate(self, X_test, y_test, classifier_name=None):
        """Evaluate a specific classifier or voting classifier"""
        
        predictions = self.predict(X_test, classifier_name)
        
        accuracy = accuracy_score(y_test, predictions)
        
        print(f"\n{'='*80}")
        print(f"EVALUATION RESULTS - {classifier_name if classifier_name else 'Voting Classifier'}")
        print(f"{'='*80}")
        print(f"Accuracy: {accuracy:.4f}\n")
        
        print("Classification Report:")
        print(classification_report(y_test, predictions, target_names=['Authentic', 'Forged']))
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, predictions))
        print(f"{'='*80}\n")
        
        return accuracy


if __name__ == "__main__":
    print("Hybrid Classifier Module - Fixed for scikit-learn 1.2+")
    print("Key changes:")
    print("  - Removed 'algorithm' parameter from AdaBoostClassifier")
    print("  - Added 'estimator' parameter instead of 'base_estimator'")
    print("  - Compatible with scikit-learn >= 1.2")
    print("  - Added get_all_predictions() method for web app")
