
def saveanddict2json(dict):
    import json
    with open('data/data.json', 'w') as fp:
        json.dump(dict, fp)



def json2dict(json_str):
    import json
    return json.loads(json_str)


data = {
    'name' : 'myname',
    'age' : 100,
}
a=saveanddict2json(data)
