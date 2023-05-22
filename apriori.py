from itertools import islice
from itertools import combinations
from collections import Counter
import csv


def read_csv(filename):
    data = []
    with open(filename, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            data.append(row)
    return data


def convert_to_list_of_lists(data):
    result = []
    for idx in range(len(data)):
        row_values = data[idx]
        row_list = [idx, row_values]
        result.append(row_list)
    return result


filename = 'store_data.csv'
csv_data = read_csv(filename)
# print(csv_data)
data = convert_to_list_of_lists(csv_data)

# print(data)

mapping = {}

init = []
for i in data:
    #     print(data)
    for q in i[1]:
        #         print(i[0])
        #         print(i[1])

        if (q not in init):
            if q == '':
                pass
            else:
                init.append(q)
init = sorted(init)
print(init)

sp = float(input("Enter the Minimum Support Value in(%): "))
sp = sp/100
conf = float(input("Enter the Minimum Confidence Value in(%): "))
s = int(sp*len(init))

# print(s)


c = Counter()
# print("c=",c)
for i in init:
    for d in data:
        if (i in d[1]):
            c[i] += 1
print("C1:")
for i in c:
    print(str([i])+": "+str(c[i]))
# print()
# print("c=",c)
l = Counter()
for i in c:
    if (c[i] >= s):
        l[frozenset([i])] += c[i]
print("L1:")
for i in l:
    print(str(list(i))+": "+str(l[i]))
print()
pl = l
pos = 1
for count in range(2, 5000):
    nc = set()
    temp = list(l)
    for i in range(0, len(temp)):
        for j in range(i+1, len(temp)):
            t = temp[i].union(temp[j])
            if (len(t) == count):
                nc.add(temp[i].union(temp[j]))
    nc = list(nc)
    c = Counter()
    for i in nc:
        c[i] = 0
        for q in data:
            temp = set(q[1])
            if (i.issubset(temp)):
                c[i] += 1
    print("C"+str(count)+":")
    for i in c:
        print(str(list(i))+": "+str(c[i]))
    print()
    l = Counter()
    for i in c:
        if (c[i] >= s):
            l[i] += c[i]
    print("L"+str(count)+":")
    for i in l:
        print(str(list(i))+": "+str(l[i]))
    print()
    if (len(l) == 0):
        break
    pl = l
    pos = count
print("Result: ")
print("L"+str(pos)+":")
for i in pl:
    print(str(list(i))+": "+str(pl[i]))
print()

for l in pl:
    c = [frozenset(q) for q in combinations(l, len(l)-1)]

    for a in c:
        b = l-a
        ab = l
        sab = 0
        sa = 0
        sb = 0
        for q in data:
            temp = set(q[1])
            if (a.issubset(temp)):
                sa += 1
            if (b.issubset(temp)):
                sb += 1
            if (ab.issubset(temp)):
                sab += 1
        temp = sab/sa*100

        temp = sab/sb*100

        print(str(list(a))+" -> "+str(list(b))+" = "+str(sab/sa*100)+"%")
        plus=["->"]
        merged_string = " ".join(list(a)+plus+list(b))
        #print(merged_string)
        mapping[merged_string]=sab/sa*100
        print(str(list(b))+" -> "+str(list(a))+" = "+str(sab/sb*100)+"%")
        merged_string = " ".join(list(b)+plus+list(a))
        #print(merged_string)
        mapping[merged_string]=sab/sb*100
    curr = 1
    print("choosing:", end=' ')
    for a in c:
        b = l-a
        ab = l
        sab = 0
        sa = 0
        sb = 0
        for q in data:
            temp = set(q[1])
            if (a.issubset(temp)):
                sa += 1
            if (b.issubset(temp)):
                sb += 1
            if (ab.issubset(temp)):
                sab += 1
        temp = sab/sa*100
        if (temp >= conf):

            print(curr, end=' ')
        curr += 1
        temp = sab/sb*100
        if (temp >= conf):

            print(curr, end=' ')
        curr += 1
    print()
    print()
N = int(input("Enter a value N to find first N number of Association rules of Highest Confidence Value in(%) : "))
sorted_map = dict(sorted(mapping.items(), key=lambda x: x[1], reverse=True))

first_N_values = dict(islice(sorted_map.items(), N))

for key, value in first_N_values.items():
    formatted_value = "{:.2f}".format(value)
    print(f"Association Rule: [{key}], Confidence Score : {formatted_value}"+"%")
