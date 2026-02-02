import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib

def train_classifier(X, y):
    # ------------------------------------------
    # Show class distribution BEFORE splitting
    # ------------------------------------------
    unique, counts = np.unique(y, return_counts=True)
    print("\n=== ORIGINAL CLASS DISTRIBUTION ===")
    for cls, cnt in zip(unique, counts):
        print(f"Class {cls}: {cnt} samples")

    # ------------------------------------------
    # 70/30 split (stratified)
    # ------------------------------------------
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=0.30,
        random_state=42,
        stratify=y
    )

    # ------------------------------------------
    # Show distribution AFTER split
    # ------------------------------------------
    print("\n=== TRAIN SPLIT DISTRIBUTION ===")
    u_train, c_train = np.unique(y_train, return_counts=True)
    for cls, cnt in zip(u_train, c_train):
        print(f"Class {cls}: {cnt}")

    print("\n=== VALIDATION SPLIT DISTRIBUTION ===")
    u_val, c_val = np.unique(y_val, return_counts=True)
    for cls, cnt in zip(u_val, c_val):
        print(f"Class {cls}: {cnt}")

    # ------------------------------------------
    # Train classifier
    # ------------------------------------------
    print("\nTraining Random Forest classifier...")
    clf = RandomForestClassifier(
        n_estimators=300,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1
    )

    clf.fit(X_train, y_train)

    print("\n=== VALIDATION RESULTS ===")
    y_pred = clf.predict(X_val)

    print("\n--- Classification Report ---")
    print(classification_report(y_val, y_pred, digits=4))

    print("\n--- Confusion Matrix ---")
    print(confusion_matrix(y_val, y_pred))

    # Feature importance
    feat_names = [
        "dn_r", "dn_g", "dn_b", "green", "red",
        "rededge", "nir", "gndvi", "lci", "ndre",
        "ndvi", "osavi", "dsm", "random_band"
    ]

    print("\n=== FEATURE IMPORTANCE ===")
    importances = clf.feature_importances_

    ranking = sorted(zip(feat_names, importances), key=lambda x: -x[1])
    for name, imp in ranking:
        print(f"{name:12s} : {imp:.6f}")

    # Save the trained model
    joblib.dump(clf, "models/classifier_rf.joblib")
    print("\nClassifier saved as classifier_rf.joblib\n")

    return clf
