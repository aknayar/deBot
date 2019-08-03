import re
import numpy as np
from gensim.models.keyedvectors import KeyedVectors
from ArgumentQualityTrain import RNN

if __name__ == "__main__":
    glove = KeyedVectors.load_word2vec_format("glove.6B.50d.txt.w2v", binary=False)
    params = np.load('ArgumentQualityModel.npy', allow_pickle=True)
    model = RNN(50, 10, 21)
    model.fc_x2h.weight, model.fc_x2h.bias, model.fc_h2h.weight, model.fc_h2y.weight, model.fc_h2y.bias, model.Uz, model.Wz, model.bz, model.Ur, model.Wr, model.br, model.Uh, model.Wh, model.bh = (
        params[0],
        params[1],
        params[2],
        params[3],
        params[4],
        params[5],
        params[6],
        params[7],
        params[8],
        params[9],
        params[10],
        params[11],
        params[12],
        params[13],
    )

    i = input("What is your argument? ")
    i = i.lower().replace("[ref]", "")
    i = "".join(c for c in i if c.isdigit() or c.isalpha() or c == " ")
    i = re.sub(r" \W+", " ", i)
    i = i.split()
    test = []
    row = []
    for word in i:
        try:
            row.append(glove[word])
        except:
            continue
    test.append(row)
    for i in test:
        for j in range(len(i), 78):
            i.append(np.zeros(50))
    w = np.ascontiguousarray(np.swapaxes(np.array(test).reshape(1,78,50),0,1))
    print(np.argmax(model(w))/20)