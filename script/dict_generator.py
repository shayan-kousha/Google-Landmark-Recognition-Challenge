import sys, csv, pickle

def label_dict_generator (csv_dict):
    label_dict = {}
    count = 0
    for dic in csv_dict:
        label_dict[dic['id']] = dic['landmark_id']
        
        count += 1
        
        if count % 100000:
            print "still working {}".format(count)
            
    return label_dict
    
if "__main__" == __name__:
    with open(sys.argv[1]) as csvfile:
        csv_dict = csv.DictReader(csvfile)
        label_dict = label_dict_generator (csv_dict)
        
        name = sys.argv[1].split(".csv")[0]
        
        with open(name + ".pkl","wb") as f:
            pickle.dump(label_dict,f)
