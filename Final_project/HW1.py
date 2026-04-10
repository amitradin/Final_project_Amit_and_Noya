"""This file is for running the K-means algorithm copied from HW1"""


import sys
EPSILON = 1e-4

def main(args):
    inputs = []
    conn = []  
    for line in sys.stdin: 
        if(line.strip()==""):
            continue
        curr = line.split(",")
        for i in range (len(curr)):
            try:
                curr[i] =float(curr[i]) #try to convert to float
            except:
                print("An Error Has Occurred")
                sys.exit(1)
        inputs.append(curr)
    if(len(inputs)<=2): # we length of the input is less than or equal to 2
        print("An Error Has Occurred")
        sys.exit(1)
    if(len(args)<2): # we didn't get k
        print("Incorrect number of clusters!")
        sys.exit(1)
    if(len(args)<3): #we didn't get the number of iterations
        iterations=400
    else:
        if(not args[2].isdigit()):
             print("Incorrect maximum iteration!")
             sys.exit(1)
        else:
            iterations=int(args[2])
            if(iterations>=800 or iterations<=1):
                print("Incorrect maximum iteration!")
                sys.exit(1)
    if(not args[1].isdigit()): # k is not an integer
        print("Incorrect number of clusters!")
        sys.exit(1)
    k= int(args[1])
    if(k<=1 or k>=len(inputs)):
        print("Incorrect number of clusters!")
        sys.exit(1)
    #Lets validate that everything is the same width
    width=len(inputs[0])
    for arr in inputs:
        if(len(arr)!=width):
            print("An Error Has Occurred")
            sys.exit(1)
    #First the centroids are the first k inputs
    Centroids = inputs[:k]
    check=False
    while(check == False and iterations>0):
        conn=calc_length_from_centroids(inputs,Centroids,k)
        temp_cen=calc_new_centroids(inputs,Centroids,conn)
        check=check_less_than_epsilon(Centroids,temp_cen)
        Centroids=temp_cen
        iterations-=1
    for centroid in Centroids:
        for i in range(len(centroid)):
            print(f"{centroid[i]:.4f}",end="")
            if(i<len(centroid)-1):
                print(',',end="")
        print("")
        
def calc_len(vector_1, vector_2):
    length=0
    for v1,v2 in zip(vector_1,vector_2):
        length+= (v1-v2)**2
    length = length**0.5
    return length

def calc_length_from_centroids(values,Centroids,k):
    conn = [0] * len(values) 
    for j,vec in enumerate(values):
        min_index= 0
        min_val= float("inf")
        for i in range(k):
            length_from_cen = calc_len(vec,Centroids[i])
            if(length_from_cen<min_val):
                min_val=length_from_cen
                min_index=i
        conn[j]=min_index
    return conn

def calc_new_centroids(values,Centroids,conn):
    sums=[0.0]*len(Centroids)
    counts=[0] *len(Centroids)
    temp_cen=[[] for i in range(len(Centroids))]
    for index in conn:
        counts[index]+=1
    for j in range(len(values[0])):
        for i in range(len(values)):
            sums[conn[i]]+=values[i][j]
        for i in range(len(sums)):
            if(counts[i]>0):
                temp_cen[i].append(sums[i]/counts[i])
            else:
                temp_cen[i].append(Centroids[i][j])
        for i in range(len(sums)): #reset sum
            sums[i]=0
    return temp_cen

def check_less_than_epsilon(Centroids,temp_cen):
    for i in range(len(Centroids)):
        if(calc_len(Centroids[i],temp_cen[i])>EPSILON):
            return False
    return True
    
                
if __name__ == "__main__":
    main(sys.argv)

   