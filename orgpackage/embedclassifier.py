from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def train_classifier_head(embeddings, labels, model):
    model.fit(embeddings, labels)
    return model
