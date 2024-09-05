import numpy as np
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score,normalized_mutual_info_score,f1_score,accuracy_score,precision_score,recall_score
import warnings
warnings.filterwarnings("ignore")


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class evaluation_metrics():
    def __init__(self, logits, labels):
        self.logits=logits
        self.labels=labels
        self.res=self.evaluate_hiPred(self.logits,self.labels)
        self.nmi=self.res['nmi']
        self.macro_f1=self.res['macro_f1']
        self.micro_f1=self.res['micro_f1']


    def evaluate_hiPred(self, logits, labels):
        logits = F.softmax(logits)
        logits = logits.cpu().data.numpy()
        labels = labels.cpu().data.numpy()
        labels_1d = labels

        logits_1d = np.argmax(logits, 1)
        result_NMI = normalized_mutual_info_score(labels_1d, logits_1d)

        macro_f1 = f1_score(labels_1d, logits_1d, average='macro')
        micro_f1 = f1_score(labels_1d, logits_1d, average='micro')



        results = { 'nmi': result_NMI, 'macro_f1': macro_f1, 'micro_f1': micro_f1}
        print(' nmi: %.4f, macro_f1: %.4f, micro_f1: %.4f' % (
         result_NMI, macro_f1, micro_f1))
        return results


