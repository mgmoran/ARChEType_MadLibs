import json

original = []
linked = []
text = []

# with open("Madlibs_Templates.jsonl", 'r') as infile:
#     for line in infile:
#         original.append(json.loads(line))

with open("Madlibs_Templates_linked.jsonl", 'r') as infile:
    for line in infile:
        linked.append(json.loads(line))

with open("All_template_text.txt", 'r') as infile:
    for line in infile:
        text.append(line.split('\n'))

for i in range(len(linked)):
    raw = text[i][0]
    tem = linked[i]["text"]
    # pl = original[i]["labels"]
    ll = linked[i]["labels"]
    ents = ""
    for j in range(len(raw)):
        if raw[j] != tem[j]:
            if raw[j] == " ":
                ents += "-"
            else:
                ents += raw[j]
        else:
            ents += " "
    el = ents.split()
    print("Data ", i)
    print("Raw text:", raw)
    print("Template:", tem)
    print("All entities", el)
    # print("Plain labels:", pl)
    print("Marked labels:", ll)
    print([(w, l) for w, l in zip(el, ll)])
    print()



