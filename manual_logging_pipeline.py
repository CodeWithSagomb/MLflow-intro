

# manual_logging_pipeline.py

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import numpy as np


mlflow.set_tracking_uri("sqlite:///mlflow.db")

# Configuration de MLflow pour le suivi de nos expériences
# L'URI pour se connecter au serveur de tracking
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Le nom de l'expérience où les runs seront enregistrés
mlflow.set_experiment("Iris Model Experiment")


def train_model(n_estimators, max_depth):
    """
    Entraîne un modèle et journalise les résultats avec MLflow.
    """
    # Démarre un nouveau run
    with mlflow.start_run(run_name=f"RF-estimators-{n_estimators}-depth-{max_depth}"):

        # --- 1. Chargement et préparation des données ---
        iris = load_iris(as_frame=True)
        X = iris.data
        y = iris.target
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # --- 2. Journalisation des Hyperparamètres ---
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)

        # --- 3. Définition des tags ---
        mlflow.set_tag("model_type", "RandomForestClassifier")
        mlflow.set_tag("dataset", "iris")

        # --- 4. Entraînement du modèle ---
        print(f"Entraînement du modèle avec n_estimators={n_estimators} et max_depth={max_depth}...")
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)

        # --- 5. Évaluation et Journalisation des métriques ---
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        mlflow.log_metric("accuracy", accuracy)
        print(f"Précision du modèle : {accuracy:.2f}")

        # --- 6. Journalisation des artefacts (graphiques) ---
        print("Génération du graphique d'importance des features...")
        importances = model.feature_importances_
        feature_names = iris.feature_names
        fig, ax = plt.subplots()
        y_pos = np.arange(len(feature_names))
        ax.barh(y_pos, importances, align='center')
        ax.set_yticks(y_pos, labels=feature_names)
        ax.invert_yaxis()
        ax.set_xlabel('Importance')
        ax.set_title('Importance des features')
        plot_path = "feature_importance.png"
        plt.savefig(plot_path)
        plt.close(fig)
        mlflow.log_artifact(plot_path)

        # --- 7. Journalisation du modèle en tant qu'artefact ---
        mlflow.sklearn.log_model(model, "model")
        print("Modèle journalisé avec succès.")


        # --- 8. Enregistrer le modèle dans le Registre de Modèles
        model_name = "IrisRandomForestClassifier"
        mlflow.register_model(
            model_uri=f"runs:/{mlflow.active_run().info.run_id}/model",
            name=model_name
        )
        print(f"Modèle enregistré sous le nom : {model_name}")


if __name__ == "__main__":
    print("--- Démarrage de l'expérimentation manuelle ---")
    
    # Lancer plusieurs runs avec des hyperparamètres différents
    train_model(n_estimators=100, max_depth=5)
    train_model(n_estimators=200, max_depth=8)
    train_model(n_estimators=150, max_depth=None)
    
    print("\n--- Expérimentation terminée ---")