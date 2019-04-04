import argparse
import numpy as np

from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

parser = argparse.ArgumentParser(description='Find optimal thresholds for fasttext output')
parser.add_argument('gold_path', help='path to gold answers')
parser.add_argument('predicted_path', help='path to predicted answers with probabilities')
parser.add_argument('task', type=int, help='subtask number: 1 or 2')
parser.add_argument('-w', action='store_true', help='subtask number: 1 or 2')

args = parser.parse_args()

def read_fasttext_prob(path):
    y_pred=[]
    for line in open(path):
        elements = line.rstrip().split(' ')
        label_probs = []
        for i in range(0, len(elements) - 1, 2):
            label, prob = elements[i:i + 2]
            label=label[9:]
            # print(label, float(prob))
            label_probs.append((label, float(prob)))
        y_pred.append([prob for label, prob in sorted(label_probs)])
    return np.array(y_pred)

y_true = np.array(list(map(int, open(args.gold_path).readlines())))
y_prob = read_fasttext_prob(args.predicted_path)

classes = sorted(set(y_true))

y_true = np.array([classes.index(kls) for kls in y_true])

print('Gold sum:', y_true.sum())
print('Zeros', np.count_nonzero(y_true==0))
# print(y_pred)

# y_prob[:, 0]+=0.5

def calc(value, classes, y_prob, kls):
    class_id=classes.index(kls)
    print('Value +', value)
    y_prob_modified = np.copy(y_prob)
    y_prob_modified[:, class_id]+=value
    y_pred = np.argmax(y_prob_modified, axis=1)
    # f1_binary = f1_score(y_true, y_pred, average='binary')
    print('Sum', y_pred.sum())
    print('Zeros', np.count_nonzero(y_pred == 0))
    if len(classes)==2:
        print('F1: %.5f \t Accuracy: %.5f' % (f1_score(y_true, y_pred, average='binary'), accuracy_score(y_true, y_pred)))
        print('Precision: %.5f \t Recall: %.5f' % (
    precision_score(y_true, y_pred, average='binary'), recall_score(y_true, y_pred, average='binary')))
        #return f1_score(y_true, y_pred, average='binary')
        #return -abs(precision_score(y_true, y_pred, average='binary')-recall_score(y_true, y_pred, average='binary'))
        return f1_score(y_true, y_pred, average='binary')-abs(precision_score(y_true, y_pred, average='binary')-recall_score(y_true, y_pred, average='binary'))
    else:
        print('F1: %.5f \t Accuracy: %.5f \t Precision: %.5f \t Recall: %.5f' % (
        f1_score(y_true, y_pred, average='micro'), accuracy_score(y_true, y_pred), precision_score(y_true, y_pred, average='micro'), recall_score(y_true, y_pred, average='micro')))
        print('F1 macro: %.5f \t Precision: %.5f \t Recall: %.5f' % (
        f1_score(y_true, y_pred, average='macro'),
        precision_score(y_true, y_pred, average='macro'), recall_score(y_true, y_pred, average='macro')))
        return f1_score(y_true, y_pred, average='micro')

from scipy.optimize import minimize

eee = lambda x: -calc(x[0], classes, y_prob,0)
x0 = [np.random.random()]
res = minimize(eee, x0, method='Powell')
print(res)
print(res['x'])

if args.w:
  with open('value','wt') as f:
    f.write(str(res['x']))

#for kls in classes:
    #class_id=classes.index(kls)
    #print('Class:', kls)
    #for value in [-0.99, -0.495, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1,0.2,0.3,0.4,0.495,0.5,0.9,0.99,0.991,0.993,0.995,0.997,0.999,0.9995,0.9999,0.99995,0.99999]:
        #calc(value, classes, y_prob, kls)

        #print()

# print(f1_score(y_true, np.array([0]*len(y_true)), average='micro'))
