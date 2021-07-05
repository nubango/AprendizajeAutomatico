import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

pd.options.mode.chained_assignment = None

df = pd.read_csv("abalone_original.csv")

new_df = df.copy()

new_df['newRings_1'] = np.where(df['rings'] <= 8,1,0)
new_df['newRings_2'] = np.where(((df['rings'] > 8) & (df['rings'] <= 10)), 2,0)
new_df['newRings_3'] = np.where(df['rings'] > 10,3,0)

new_df['newRings'] = new_df['newRings_1'] + new_df['newRings_2'] + new_df['newRings_3']

x_data = new_df.drop(['newRings','rings','sex','newRings_1','newRings_2','newRings_3'], axis = 1)
x_data.info()
y = new_df['newRings']

x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data)).values
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state=0)

def SVM_KernelLinear(x_train, y_train, x_test, y_test, C):
    svm = SVC(C=C, kernel='linear', tol=1e-3, max_iter=-1)
    svm.fit(x_train, y_train)
    accuracy = svm.score(x_test, y_test)

    print("Porcentaje de acierto (C = {}): {:.2f}%".format(C, accuracy * 100))
    return accuracy


def prueba():
    cs = np.arange(1, 302, 10)
    percentages = np.zeros(len(cs))

    for i in range(len(cs)):
        percentages[i] = SVM_KernelLinear(x_train, y_train, x_test, y_test, cs[i]) * 100

    plt.plot(cs, percentages, c="orange")
    plt.xlabel("C")
    plt.ylabel("% de acierto")
    plt.show()

# Realizamos experimentos individuales con C = 11, 21, 31 ..... 201
prueba()
