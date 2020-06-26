import matplotlib.pyplot as plt

res =[]
with open("./score.txt","r") as f:
    for line in f:
        res.append(line.split())
labels=[]
with open("./tnews_public/labels.json",'r') as f:
    for line in f:
        a= eval(line)
        labels.append(a["label_desc"][5:])
p=[]
r=[]
f=[]
for i in res[1:16]:
    str = float(i[1])
    p.append(str)
    r.append(float(i[2]))
    f.append(float(i[3]))
plt.xlabel("label description")
plt.ylabel("score")
plt.plot(labels, p, label='precision')
plt.plot(labels, r, label='recall')
plt.plot(labels, f, label='f1')
plt.axhline(y=0.5,color='r',label='p_macro_avg')
plt.axhline(y=0.44,color='g',label='r_macro_avg')
plt.axhline(y=0.45,color='b',label='f_macro_avg')
plt.legend()
plt.show()