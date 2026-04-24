"""
Shadow Clone Jutsu - Gesture Model Trainer
Trains a classifier to recognize the cross-hand sign
"""

import json
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import os
import glob

class GestureTrainer:
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.data_dir = "hand_sign_data"
        self.model_path = "gesture_model.pkl"
        
    def load_training_data(self):
        """Load all collected training samples"""
        all_samples = []
        
        # Find all dataset files
        dataset_files = glob.glob(os.path.join(self.data_dir, "dataset_*.json"))
        
        if not dataset_files:
            print("❌ No training data found!")
            print(f"Please run collect_hand_data.py first to collect samples.")
            return None, None
        
        print(f"\n📂 Found {len(dataset_files)} dataset file(s)")
        
        # Load all samples
        for filepath in dataset_files:
            with open(filepath, 'r') as f:
                data = json.load(f)
                all_samples.extend(data)
        
        print(f"📊 Total samples loaded: {len(all_samples)}")
        
        # Extract features and labels
        X = []
        y = []
        
        for sample in all_samples:
            X.append(sample['features'])
            y.append(1 if sample['label'] == 'shadow_clone' else 0)
        
        return np.array(X), np.array(y)
    
    def generate_negative_samples(self, X, n_samples=None):
        """Generate random negative samples (non-gesture hand positions)"""
        if n_samples is None:
            n_samples = len(X) // 2  # Generate 50% negative samples
        
        print(f"🔄 Generating {n_samples} negative samples...")
        
        # Create random hand positions (noise)
        feature_dim = X.shape[1]
        negative_samples = np.random.uniform(0, 1, (n_samples, feature_dim))
        
        return negative_samples
    
    def train(self):
        """Train the gesture recognition model"""
        print("=" * 60)
        print("🥷 SHADOW CLONE JUTSU - MODEL TRAINING")
        print("=" * 60)
        
        # Load data
        X_positive, y = self.load_training_data()
        
        if X_positive is None:
            return False
        
        # Generate negative samples
        X_negative = self.generate_negative_samples(X_positive)
        y_negative = np.zeros(len(X_negative))
        
        # Combine positive and negative samples
        X = np.vstack([X_positive, X_negative])
        y = np.hstack([np.ones(len(X_positive)), y_negative])
        
        print(f"\n📈 Training dataset:")
        print(f"   - Positive samples (gesture): {len(X_positive)}")
        print(f"   - Negative samples (no gesture): {len(X_negative)}")
        print(f"   - Total samples: {len(X)}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"\n🔧 Training model...")
        print(f"   - Training set: {len(X_train)} samples")
        print(f"   - Test set: {len(X_test)} samples")
        
        # Train
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n✅ Training complete!")
        print(f"   - Accuracy: {accuracy * 100:.2f}%")
        
        print("\n📊 Classification Report:")
        print(classification_report(y_test, y_pred, 
                                   target_names=['No Gesture', 'Shadow Clone']))
        
        # Save model
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        print(f"\n💾 Model saved to: {self.model_path}")
        
        if accuracy < 0.85:
            print("\n⚠️  Warning: Accuracy is below 85%")
            print("   Consider collecting more training samples for better performance.")
        else:
            print("\n🎉 Model trained successfully!")
            print("   You can now run the main shadow_clone_jutsu.py application!")
        
        return True

if __name__ == "__main__":
    trainer = GestureTrainer()
    trainer.train()
