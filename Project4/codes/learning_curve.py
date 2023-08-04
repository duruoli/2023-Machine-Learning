import matplotlib.pyplot as plt
import pandas as pd

df1 = pd.read_csv('Linear.csv')
df2 = pd.read_csv('VanillaCNN2.csv').iloc[:20]
df3 = pd.read_csv('VanillaCNN3.csv').iloc[:20]

attrs = ['Train Accuracy', 'Validation Accuracy', 'Train Loss', 'Validation Loss']

for attr in attrs:
    # Assuming you have three lists of training losses or accuracies
    train_loss_curve_1 = df1[attr]
    train_loss_curve_2 = df2[attr]
    train_loss_curve_3 = df3[attr]

    # Assuming you have the corresponding list of epoch numbers
    epochs = df1.index.values

    # Plotting the learning curves
    plt.plot(epochs, train_loss_curve_1, label='Linear')
    plt.plot(epochs, train_loss_curve_2, label='VanillaCNN3_SGD')
    plt.plot(epochs, train_loss_curve_3, label='VanillaCNN2_Adam')

    # Adding labels and title
    plt.xlabel('Epochs')
    plt.ylabel(attr)
    plt.title('Learning Curves:'+attr)

    # Adding a legend
    plt.legend()

    # Displaying the plot
    plt.show()
