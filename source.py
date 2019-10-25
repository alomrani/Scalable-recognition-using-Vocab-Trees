import time
import numpy as np
from skimage import data
from skimage import io
from skimage import color
from skimage import data_dir
from scipy import ndimage
import math
import imageio
from skimage import transform as tf
from skimage import filters
from skimage.feature import match_template
from skimage.feature import peak_local_max
from skimage.transform import pyramid_gaussian
import matplotlib.pyplot as plt
import cv2
%pylab inline
import os
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin


VocabTree = {} #Vocab Tree
NumBranches = 10 #Branching factor
leaves = {} # Stores cluster centres of each leaf
dvdCoversPath = "C:/Users/omran/desktop/project4/project4/DVDcovers/"  
descriptor = {}
deslist = [] #list of descriptors of all images
max_level = 4 # depth of vocab tree
sift = cv2.xfeatures2d.SIFT_create()
numOfDescriptors = 0
#Bag of words list for each database image
BoGs = {}
getAllTrainDescriptors(dvdCoversPath) #dump all descriptors to desList
print("descriptors extracted")
BuildTree(0, deslist, 0)
print("Tree built")
InvertedFile = {}
#stores all images with a descriptor in leaf i
N = {}
temp  = {}
#stores all leafs containing descritpors for image i
n = {}
#######Initialize inverted file and other data structures#######
imageNames = os.listdir(dvdCoversPath)
#loop through all files except .DS_STORE file
for imageName in imageNames[1:]:
    if '.ini' not in imageName:
        temp[imageName] = 0
        n[imageName] = {}
for i in leaves.keys():
    InvertedFile[i] = temp.copy()
    N[i] = temp.copy()
#Build Inverted File Index and Bag of words list for each database Image
InvertedFileIndex(dvdCoversPath)
print("Inverted File index built")




queryImagePath= "C:/Users/omran/desktop/project4/project4/test/test2.png"
q = computeBagofWords(queryImagePath, dvdCoversPath)
#Sort by score
sortedBogs = sorted(BoGs.items(), key=lambda kv: ComputeScore(kv[1],q), reverse=False)
#retrieve top 10 database images
top_10 = [i[0] for i in sortedBogs[:20]] 
print("Top Matches found")
print(top_10)



def getAllTrainDescriptors(path):
    """
    Given the path to all images, 
    dumps all descriptors into one big list to be used in clustering.
    """
    global deslist, descriptor, numOfDescriptors
    imageNames = os.listdir(path)
    #Ignore .DS_STORE file
    for imageName in imageNames[1:]:
        #Ignoring windows desktop.ini file
        if ".ini" not in imageName:
            image = cv2.imread(path + "/" + imageName)   
            image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            kp, des = sift.detectAndCompute(image, None)
            descriptor[imageName] = des
            deslist.extend(des)
            numOfDescriptors += len(des)
            
            
            
  
  
def BuildTree(level, deslist, node):
    """
    Given the current level, the list of descriptors to cluster, and the current node being built,
    builds the children of node if max level is not reached.
    """
    global NumBranches, clustering, VocabTree, leaves
    clustering = KMeans(n_clusters=NumBranches, max_iter=500)
    if (level == max_level):
        return
    #Cluster the list of descritpors
    clustering.fit(deslist)
    centres = clustering.cluster_centers_
    clusters = [[] for i in range(NumBranches)]
    #Appending each descriptor to its cluster
    for i in range(len(deslist)):
        clusters[clustering.labels_[i]].append(deslist[i])
    if level == max_level - 1:
        #assign the leaves to the clusters and
        #store the cluster centres in another dictionary 
        for i in range(NumBranches):
            VocabTree[node*NumBranches + 1 + i] = clusters[i]
            leaves[node*NumBranches + 1 + i] = centres[i]
    else:
        #Each node stores its cluster centre, except for leaf nodes
        for i in range(NumBranches):
            VocabTree[node*NumBranches + 1 + i] = centres[i]
    #Build the children subTrees
    for j in range(NumBranches):
        BuildTree(level+1, clusters[i], node*NumBranches+j+1)


def isLeaf(nodeIndex):
    """
    Returns True if node at nodeIndex is a leaf node, False otherwise
    """
    global max_level, NumBranches
    #Total number of nodes in a full tree
    numNodes = (((NumBranches)**(max_level+1) - 1) / (NumBranches-1))
    #Total number of leaf nodes is a full tree
    numLeaves = NumBranches**(max_level)
    return nodeIndex >= numNodes - numLeaves
    
    
def findLeaf(des, currNode):
    """
    Given the descriptor and the current node in the tree, 
    lookup the descriptor in the tree util leafnode is reached.
    """
    children = [currNode*NumBranches + i + 1 for i in range(NumBranches)]
    #Leaf node reached
    if isLeaf(currNode):
        return currNode
    else:
        distances = []
        #leaf nodes in the tree store clusters, must use leaves dictionary to access cluster centres of leaf nodes
        if isLeaf(children[0]):
            distances = [float(np.dot(leaves[child], des))/float((np.linalg.norm(leaves[child])*np.linalg.norm(des))) for child in children]
        else:
            #distance of each cluster centre from des
            distances = [float(np.dot(VocabTree[child], des))/float((np.linalg.norm(VocabTree[child])*np.linalg.norm(des))) for child in children]
        #recursive lookup on cluster with closest centre
        return findLeaf(des, children[np.argmax(distances)])
       
       
       
       
def InvertedFileIndex(path):
    imageNames = os.listdir(path)
    global InvertedFile
    for imageName in imageNames[1:]:
        if '.ini' not in imageName:
            for des in descriptor[imageName]:
                #Find the word which contains des
                leaf = findLeaf(des, 0)
                #update each dictionary
                n[imageName][leaf] = 1
                InvertedFile[leaf][imageName] += 1
                N[leaf][imageName] = 1
        
    #Build BoG list for each image
    for imageName in imageNames[1:]:
        if '.ini' not in imageName:
            BoG = []
            for i in leaves.keys():
                #if a leaf contains no descriptors, give it weight 0
                if sum(N[i].values()) == 0:
                     BoG.append(0.0)
                else:
                    #same weight formula proposed in paper
                    BoG.append((float(InvertedFile[i][imageName])/float(len(n[imageName].keys()))) * (np.log10(float(len(imageNames[1:]))/float(sum(N[i].values())))))
            BoGs[imageName] = BoG[:]
  
  
  
  
  
 def computeBagofWords(queryImagepath, path):
    """
    Compute bag of words list for query image, given path of query image and path of database images
    
    """
    imageNames = os.listdir(path)
    queryImage = cv2.imread(queryImagepath)
    queryImage = cv2.cvtColor(queryImage,cv2.COLOR_BGR2GRAY)
    kp, des = sift.detectAndCompute(queryImage, None)
    #Stores the number of descripors that query image has in leaf i
    tempFileIndex = {}
    #stores all leafs which have descriptors of query image
    k = {}
    for i in leaves.keys():
        tempFileIndex[i] = 0
    for desc in des:
        leaf = findLeaf(desc, 0)
        tempFileIndex[leaf] += 1
        k[leaf] = 0
    BoG = []
    t = len(k.keys())
    #Build BoG list of query image
    for i in leaves.keys():
        if sum(N[i].values()) == 0:
            BoG.append(0.0)
        else:
            BoG.append((float(tempFileIndex[i])/float(t))* np.log10(float(len(imageNames[1:]))/float(sum(N[i].values()))))
    return BoG
   
   
   
   
   
   
def ComputeScore(d, q):
    """
    returns score given query BoG list and database BoG list
    """
    d1 = np.linalg.norm(d,1)
    q1 = np.linalg.norm(q,1)
    return np.linalg.norm(d/d1 - q/q1, 1)
    
    
    
fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(20, 25))
#best homography matrix
best_homography = 0
#max in-liners across all images
max_inliners_allimages = 0
#most matching image
best_image = 0
landscape_2 = cv2.imread(queryImagePath)
landscape_2= cv2.cvtColor(landscape_2,cv2.COLOR_BGR2GRAY)
for image in top_10:
    landscape_1 = cv2.imread(dvdCoversPath + image)
    landscape_1= cv2.cvtColor(landscape_1,cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    kp, des = sift.detectAndCompute(landscape_1,None)
    kp1, des1 = sift.detectAndCompute(landscape_2,None)
    #Using brute force builtin matching
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des,k=2)
    good = []
    #Applying ratio test
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append(m)
    matches = good
    max_inliners = 0
    max_H = []
    for i in range(1000):
        #Pick 4 random matches
        np.random.shuffle(matches)
        p = matches[0:4]
        #Get x,y coordinates of matches
        x = [kp[p[i].trainIdx].pt[0] for i in range(len(p))]
        y = [kp[p[i].trainIdx].pt[1] for i in range(len(p))]
        x1 = [kp1[p[i].queryIdx].pt[0] for i in range(len(p))]
        y1 = [kp1[p[i].queryIdx].pt[1] for i in range(len(p))]
        #Compute A matrix
        A = np.array([[x[0], y[0], 1, 0, 0, 0, -x1[0]*x[0], -x1[0]*y[0], -x1[0]],
             [0,0,0,x[0], y[0], 1, -y1[0]*x[0], -y1[0]*y[0], -y1[0]],
             [x[1], y[1], 1, 0, 0, 0, -x1[1]*x[1], -x1[1]*y[1], -x1[1]],
             [0,0,0,x[1], y[1], 1, -y1[1]*x[1], -y1[1]*y[1], -y1[1]],
             [x[2], y[2], 1, 0, 0, 0, -x1[2]*x[2], -x1[2]*y[2], -x1[2]],
             [0,0,0,x[2], y[2], 1, -y1[2]*x[2], -y1[2]*y[2], -y1[2]],
             [x[3], y[3], 1, 0, 0, 0, -x1[3]*x[3], -x1[3]*y[3], -x1[3]],
             [0,0,0,x[3], y[3], 1, -y1[3]*x[3], -y1[3]*y[3], -y1[3]]])
        #Solve by finding eigne vector with smallest eigen value
        h = np.linalg.eig(A.T.dot(A))[1][:, np.argmin(np.linalg.eig(A.T.dot(A))[0])]
        #Homography matrix
        H = np.array([h[:3],
             h[3:6],
             h[6:9]])
        #Warp all matches with matrix
        warped = H.dot(np.row_stack(([kp[matches[i].trainIdx].pt[0] for i in range(len(matches))],[kp[matches[i].trainIdx].pt[1] for i in range(len(matches))],[1]*(len(matches))))) 
        warped = warped / warped[2,:]
        xt = [kp1[matches[i].queryIdx].pt[0] for i in range(len(matches))]
        yt = [kp1[matches[i].queryIdx].pt[1] for i in range(len(matches))]
        cur = 0
        #Find num inliners between all matches
        for j in range(len(matches)):
            if (np.sqrt((warped[0][j] - xt[j])**2 + (warped[1][j] - yt[j])**2) < 2):
                cur += 1
        if (cur > max_inliners):
            max_inliners = cur
            max_h = H    
    #Warp using best homography matrix with heighest inliners
    #dst = cv2.warpPerspective(landscape_2, max_h,(landscape_2.shape[1], landscape_2.shape[0]))
    if (max_inliners > max_inliners_allimages):
            max_inliners_allimages = max_inliners
            best_homography = max_h
            best_image = image
best_image = cv2.imread(dvdCoversPath + best_image)
query_image=cv2.imread(queryImagePath)
best_image= cv2.cvtColor(best_image,cv2.COLOR_BGR2GRAY)
query_image = cv2.cvtColor(query_image,cv2.COLOR_BGR2GRAY)
#Warp using best homography matrix with heighest inliners
dst = tf.warp(landscape_2, best_homography, output_shape=(best_image.shape[0], best_image.shape[1]))
#warped localized image
ax1.imshow(dst, cmap="gray")
#matching DVD cover
ax2.imshow(best_image, cmap="gray")
#Query Image
ax3.imshow(query_image, cmap="gray")
    
