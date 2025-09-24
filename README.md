# MLflow Iris Classifier Project

Welcome to this little project on Machine Learning Operations (MLOps) using **MLflow**. This repository is a sandbox where I've been learning to productionize a simple classification model.

### The Goal 🎯
The main objective of this project was to understand the core principles of MLOps, specifically:
-   **Experiment Tracking**: Logging different model runs, parameters, and metrics.
-   **Model Registry**: Managing the lifecycle of a model (versions, stages like Staging or Production).
-   **Model Serving**: Deploying the model as a simple REST API for real-time inference.

I've used the classic **Iris dataset** and a **Random Forest Classifier** as a hands-on example to make these concepts concrete.

### What's Inside? 📦
-   `manual_logging_pipeline.py`: A Python script that trains the model and manually logs its parameters, metrics, and the model itself to the MLflow tracking server.
-   `autologging_pipeline.py`: An alternative script that shows how MLflow's `autologging` feature can simplify the logging process, handling most of the work for you.
-   `mlflow.db`: This is our local SQLite database. It's the central hub for all our experiment tracking data and model registry metadata.
-   `mlruns/` and `mlartifacts/`: These directories are created by MLflow to store artifacts from our experiments, like the serialized models, plots, and other files.
-   `iris_input.json`: A sample file with input data to test the deployed model API.

### How to Run It? 🚀

First things first, make sure you have a working Python environment (I used a virtual environment).

1.  **Clone the repository**
    ```bash
    git clone [https://github.com/votre_nom_utilisateur/MLflow-intro.git](https://github.com/votre_nom_utilisateur/MLflow-intro.git)
    cd MLflow-intro
    ```
2.  **Activate your virtual environment**
    ```bash
    .\venv\Scripts\activate  # For Windows
    source venv/bin/activate # For macOS/Linux
    ```
3.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: You might need to generate this file first with `pip freeze > requirements.txt`)*

4.  **Launch the MLflow UI**
    This command starts the local MLflow tracking server, which acts as our central hub.
    ```bash
    mlflow ui --backend-store-uri sqlite:///mlflow.db
    ```
    Now, open your browser and go to `http://127.0.0.1:5000` to see your experiments live!

5.  **Run the training script**
    In a separate terminal, run the pipeline to train and log the model.
    ```bash
    python manual_logging_pipeline.py
    # or
    python autologging_pipeline.py
    ```

6.  **Deploy the model as an API**
    Once you've registered a model, you can serve it in a third terminal. This creates a REST API for real-time predictions. Make sure the MLflow UI server is still running.
    ```bash
    mlflow models serve -m "models:/IrisRandomForestClassifier/3" --no-conda --port 5001
    ```
    (Note: The version number `3` might change depending on your runs.)

7.  **Test the API**
    Finally, you can send data to the API to get a prediction back.
    ```bash
    curl.exe -X POST -H "Content-Type: application/json" --data "@iris_input.json" [http://127.0.0.1:5001/invocations](http://127.0.0.1:5001/invocations)
    ```
    You should get a result like `{"predictions": [0]}` back!

### What's Next?
Feel free to fork this project, play with the code, and try different models or parameters. It's a great starting point for anyone looking to get their hands dirty with MLflow and MLOps.

Happy coding! 