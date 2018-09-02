import csv

def adad():
    with open("../data/submission.csv", 'r') as csvfile:
        reader = csv.reader(csvfile)
        
        count = 0
        
        for row in reader:
            if count:
                int(row[1])
            count += 1
def example():
    with open("../../submission.csv", 'r') as csvfile:
        reader = csv.reader(csvfile)
        
        with open("../data/example.csv", 'w') as f:
            writer = csv.writer(f, delimiter=',',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
            
            writer.writerow(['id', 'landmarks'])
            
            for row in reader:
                writer.writerow([str(row[0]), str(9633) + " " + str(0.8)])
def fill():
    reader = None
    with open("../data/missing.csv", 'r') as csvfile:
        reader = csv.reader(csvfile)
        
        with open("../../submission.csv", "a+") as f:
            writer = csv.writer(f, delimiter=',',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
            
            for row in reader:
                writer.writerow([str(row[0]), str(9633) + " " + str(0.5)])
        
def find_missing ():
    data_list = []
        
    with open("../data/test.csv", "r") as f:
        data = csv.reader(f)
        for row in data:
            data_list.append(str(row[0]))
    
    with open("../../submission.csv", "r") as ff:
        data = csv.reader(ff)
        for row in data:
            try:
                data_list.remove(str(row[0]))
            except ValueError:
                pass
            
    with open("../data/missing.csv", 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        
        #writer.writerow(['id', 'landmarks'])
        
        for im in range(len(data_list)):
            writer.writerow([str(data_list[im]), str(9633)])
            
        
if __name__ == "__main__":
    #find_missing()
    #fill()
    example()
    #adad()