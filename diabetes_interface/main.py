import os.path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.tree import DecisionTreeClassifier
import pickle
import tkinter as tk
from sklearn.preprocessing import MinMaxScaler



#Из предварительно построенной матрицы корреляции видно, 
#что самая высокая корреляция между результатом и глюкозой, индексом массы тела, возрастом и инсулином/
#Выберем эти параметры для обучения модели


if not os.path.exists('clf.sav'):
    data = pd.read_csv("diabetes.csv")
    target_name='Outcome'
    #Предобработка данных
    #Заменяем пропущенные значения на средние
    data[["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]] = data[["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]].replace(0, np.NaN) 
    data["Glucose"].fillna(data["Glucose"].mean(), inplace = True)
    data["Insulin"].fillna(data["Insulin"].mean(), inplace = True)
    data["BMI"].fillna(data["BMI"].mean(), inplace = True)

    #data=data[["Glucose", "Insulin", "BMI", "Age", "Outcome"]]
    columns=["Glucose", "Insulin", "BMI", "Age"]
    #Нормализация признаков (от 0 до 1)
    sc = MinMaxScaler(feature_range = (0, 1))
    scaled_data=sc.fit_transform(data[columns])
    pickle.dump(sc, open('scaler.sav', 'wb'))
    scaled_data = pd.DataFrame(scaled_data, columns=columns)
    data=pd.concat([scaled_data, data[target_name]], axis=1)


    X_train, X_test, y_train, y_test = train_test_split(
    data[["Glucose", "Insulin", "BMI", "Age"]],
    data[target_name],
    test_size=0.3,
    random_state=0,
    shuffle=True)

    #Нормализация дисбаланса классов в тренировочной выборке
    horizontal_concat = pd.concat([X_train, y_train], axis=1)
    data_1 = horizontal_concat.loc[(horizontal_concat['Outcome']== 1)]
    data_0 = horizontal_concat.loc[(horizontal_concat['Outcome']== 0)]
    vertical_concat = pd.concat([data_1, data_0[:180]], axis=0)
    vertical_concat = shuffle(vertical_concat, random_state=0)
    X_train=vertical_concat.drop(target_name, axis=1)
    y_train=vertical_concat[target_name]

    #После предварительной оценки выбрана модель и параметры
    model = DecisionTreeClassifier(random_state=0,max_depth= 5, min_samples_leaf=3, min_samples_split=2)

    #Обучение модели
    model.fit(X_train, y_train)

    pickle.dump(model, open('clf.sav', 'wb'))

    


def result(prediction):
    window = tk.Tk()
    window.title("Результат")
    window.geometry("250x200")
    window.eval('tk::PlaceWindow . center') 
    if prediction==1:
        bg_color='#D37C95'
        txt_color='#400A4E'
        window.config(bg=bg_color)

        space_label = tk.Label(window, text="",  font=("Arial", 25), bg=bg_color , fg=txt_color)
        space_label.pack()

        warning_label = tk.Label(window, text="Diabetes",  font=("Arial", 25) , bg=bg_color, fg=txt_color)
        warning_label.pack()
    else:
        bg_color='#8DAC36'
        txt_color='#400A4E'
        window.config(bg=bg_color)

        space_label = tk.Label(window, text="",  font=("Arial", 25), bg=bg_color , fg=txt_color)
        space_label.pack()

        ok_label = tk.Label(window, text="No diabetes",  font=("Arial", 25) , bg=bg_color, fg=txt_color)
        ok_label.pack()

    heart_label = tk.Label(window, text="♥ ♥ ♥",  font=("Arial", 40), bg=bg_color , fg=txt_color)
    heart_label.pack()
    return

def make_prediction():
    glucose = glucose_entry.get() if glucose_entry.get()!='' else 0
    insulin = insulin_entry.get() if insulin_entry.get()!='' else 0
    bmi = bmi_entry.get() if bmi_entry.get()!='' else 0
    age = age_entry.get() if age_entry.get()!='' else 0
    
    clf = pickle.load(open('clf.sav', 'rb'))
    loaded_scaler = pickle.load(open('scaler.sav', 'rb'))
    columns=["Glucose", "Insulin", "BMI", "Age"]
    input_data=pd.DataFrame(data=[[glucose, insulin, bmi, age]], columns=columns)
    scaled_data=loaded_scaler.transform(input_data)
    
    scaled_data=pd.DataFrame(scaled_data, columns=columns)
    prediction = clf.predict(scaled_data)
    prediction = int(prediction[0])

    result(prediction)

    return



#Интерфейс
root = tk.Tk()

root.title("Предсказатель диабета")
root.geometry("700x500")
root.eval('tk::PlaceWindow . center') 


bg_color='#0D4261'
txt_color='#B99BBD'

root.config(bg=bg_color)




glucose_label = tk.Label(root, text="Glucose",  font=("Arial", 25) , bg=bg_color, fg=txt_color)
glucose_label.pack()
glucose_entry = tk.Entry(root, width=20, font=('Arial', 25))
glucose_entry.pack()

insulin_label = tk.Label(root, text="Insulin",  font=("Arial", 25),   bg=bg_color, fg=txt_color)
insulin_label.pack()
insulin_entry = tk.Entry(root, width=20, font=('Arial', 25))
insulin_entry.pack()

bmi_label = tk.Label(root, text="BMI",  font=("Arial", 25) , bg=bg_color, fg=txt_color)
bmi_label.pack()
bmi_entry = tk.Entry(root, width=20, font=('Arial', 25))
bmi_entry.pack()

age_label = tk.Label(root, text="Age",  font=("Arial", 25), bg=bg_color , fg=txt_color)
age_label.pack()
age_entry = tk.Entry(root, width=20, font=('Arial', 25))
age_entry.pack()

space_label = tk.Label(root, text="",  font=("Arial", 25), bg=bg_color , fg=txt_color)
space_label.pack()

predict_button = tk.Button(root, text="Predict", font=("Arial", 25), command=make_prediction, bg=txt_color)
predict_button.pack()

heart_label = tk.Label(root, text="\n♥ ♥ ♥ ♥ ♥ ♥",  font=("Arial", 30), bg=bg_color , fg=txt_color)
heart_label.pack()


root.mainloop()