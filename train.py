import joblib
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

wine = load_wine(as_frame=True)
X = wine.data.rename(columns={"od280/od315_of_diluted_wines": "od280_od315_of_diluted_wines"})
y = wine.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=5)),
    ("modelo", LogisticRegression(max_iter=1000)),
])

pipe.fit(X_train, y_train)
print(pipe)
print("Accuracy en test:", pipe.score(X_test, y_test))

joblib.dump(pipe, "pipeline.joblib")

pipe_cargado = joblib.load("pipeline.joblib")
print(pipe_cargado.predict(X_test.iloc[:5]))
