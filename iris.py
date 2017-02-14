import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import tensorflow as tf


# reading the dataset
df = pd.read_csv('iris.csv')


# visualization using matplotlib
plt.subplot(2, 1, 1)
for key, val in df.groupby('species'):

    plt.plot(val['sepal length'], val['sepal width'],
             label=key, marker='.', ls='')

plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')


plt.subplot(2, 1, 2)
for key, val in df.groupby('species'):

    plt.plot(val['petal length'], val['petal width'],
             label=key, marker='.', ls='')

plt.xlabel('Petal Length')
plt.ylabel('Petal Width')

plt.legend(loc='best')
# uncomment the following line to view the distribution
# plt.show()

# preprocessing the dataset
X = df.loc[:, ('sepal length', 'sepal width',
               'petal length', 'petal width')].as_matrix()

le = LabelEncoder()
ohe = OneHotEncoder()
y = ohe.fit_transform(le.fit_transform(df['species']).reshape(-1, 1)).toarray()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)


inp = tf.placeholder(tf.float32, [None, 4], name='inp')
weights = tf.Variable(tf.zeros([4, 3]))
bias = tf.Variable(tf.zeros([3]))

logits = tf.matmul(inp, weights) + bias
expected = tf.placeholder(tf.float32, [None, 3], name="expected")

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, expected, name='xentropy')
loss = tf.reduce_mean(cross_entropy)

train_step = tf.train.AdamOptimizer(0.01).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(expected, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

tf.summary.scalar('Cross Entropy Loss', loss)
tf.summary.scalar('Accuracy', accuracy)
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("./logs")


init = tf.global_variables_initializer()

with tf.Session() as sess:

    sess.run(init)

    print("Training")

    for i in range(5000):
        sess.run(train_step, feed_dict={inp: X_train,
                                        expected: y_train})

        summary, acc, err = sess.run([merged, accuracy, loss], feed_dict={inp: X_train,
                                                                           expected: y_train})

        writer.add_summary(summary, i + 1)

        if (i + 1) % 1000 == 0:
            print("Epoch: {:5d}\tAcc: {:6.2f}%\tErr: {:6.2f}".format(i + 1, acc * 100, err))


    print("\nValidation")
    acc = sess.run(accuracy, feed_dict={inp: X_test,
                                        expected: y_test})

    print("Accuracy = {:6.2f}%".format(acc * 100))
