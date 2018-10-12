# -*- coding: utf-8 -*-
"""
Define Threshold for Answer Policy

AUTHOR: Yue Peng
EMAIL: yuepeng@sf-express.com
DATE: 2018.10.03
"""
# TODO
# Define threshold
model.load_state_dict(torch.load(config.ini["modelDir"]+"/model_0.8633_lstm.pt", map_location="cpu"))
model.eval()
for x, y, lens in test_dl:
    x = torch.tensor(x).to(device)
    y = torch.tensor(y).to(device)
    lens = torch.tensor(lens).to(device)
    logits = model(x, lens)
    pred = logits.argmax(dim=-1)
    num_right_test = (pred == y).sum().item()
    num_total_test = x.size(0)
    acc = num_right_test / float(num_total_test)

result = logits.data.numpy()
y = y.data.numpy()
zeroone = np.equal(np.argmax(result,1), y)
idx_correct = np.where(zeroone==True)[0]
top3_scores_correct = []
for i in idx_correct:
    top3_scores_correct.append(sorted(result[i].tolist(),reverse=True)[:3])
d = []
for l in top3_scores_correct:
    d.append(l[0]-l[1])

top1_score = np.max(result, 1)
correct_score = np.max(result, 1)[np.where(zeroone==True)]

cnt = 0
top1_cnt = 0
top3_cnt = 0
top1_acc = 0
top3_acc = 0
filter1_idx = []
for i, s in enumerate(result):
    if np.max(s) >= 22:
        filter1_idx.append(i)
        # top1_acc: 0.9677708, top1_ratio: 0.4247148
        top1_cnt += 1
        if np.argmax(s) == y[i]:
            cnt += 1
            top1_acc += 1
    else:
        # top1_acc: 0.9104, top1_ratio: 0.6832
        # top3_acc: 0.8511, top3_ratio: 0.3167
        # overall_acc: 0.8916
        top3_scores = sorted(s.tolist(), reverse=True)[:3]
        if top3_scores[0] - top3_scores[1] >= 4:
            top1_cnt += 1
            if s.tolist().index(top3_scores[0]) == y[i]:
                cnt += 1
                top1_acc += 1
        elif top3_scores[0] - top3_scores[1] < 0.4:
            top1_cnt += 1
            if s.tolist().index(top3_scores[1]) == y[i]:
                cnt += 1
                top1_acc += 1
        else:
            top3_cnt += 1
            for score in top3_scores:
                if s.tolist().index(score) == y[i]:
                    cnt += 1
                    top3_acc += 1
                    break
idx_rest = [i for i in range(2630) if i not in filter1_idx]
result_after_filter1 = result[idx_rest, :]

top3_scores= []
for i in range(result_after_filter1.shape[0]):
    top3_scores.append(sorted(result_after_filter1[i].tolist(), reverse=True)[:3])
d = []
for l in top3_scores:
    d.append(l[0] - l[1])
d2 = []
for l in top3_scores:
    d2.append(l[1] - l[2])

idx2 = np.where(np.array(d)>=4)[0]
np.argmax(result_after_filter1,1)



def get_index(row):
    s = row.tolist()
    idx = sorted(range(len(s)), key=lambda k: s[k], reverse=True)
    return idx[:3]

top3_ans = []
for i in range(result.shape[0]):
    top3_ans.append(get_index(result[i]))

top3_correct = 0
for i, l in enumerate(y):
    if l in top3_ans[i]:
        top3_correct += 1