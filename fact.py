import scipy.io
import pylab as plt
import math
import numpy as np
import difflib
import pickle
from matplotlib.patches import Ellipse
#from nltk import ngrams

#import numpy as np  
import scipy
#import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances #jaccard diss.
from sklearn import manifold  # multidimensional scaling
from sklearn.mixture import GaussianMixture


class BoxAnalysis():

      def __init__(self):
          pass

      #def calculate_overlapping(s1,s2):
      #    f2 = s2[0]

      #least common subsequence p.287 of Sedgewick and Wayne
      def lcs(self,s,t):
          m = len(s)
          n = len(t)

          opt = np.zeros((m+1,n+1),dtype = int)
          for i in range(m-1,-1,-1):
              for j in range(n-1,-1,-1):
                  if s[i] == t[j]:
                     opt[i,j] = opt[i+1,j+1] + 1
                  else:
                     opt[i,j] = np.max((opt[i+1,j],opt[i,j+1]))

          lcs = ""
          i = 0
          j = 0
          
          while ((i<m) and (j<n)):
          	if (s[i] == t[j]):
                   lcs += s[i]
                   i+=1
                   j+=1
                elif (opt[i+1,j]>=opt[i,j+1]):
                   i+=1
                else:
                   j+=1
	  	
	  return lcs

      #must be DNA strings
      def determine_2_shingles(self,mrf,cycle_label): 
          shingles = [] 
          for k in range(0,len(mrf)-1):
              shingles.append(mrf[k:k+2])
          if cycle_label == 1:
             shingles.append(mrf[-1]+mrf[0])
          return shingles

      #must be DNA strings
      def determine_Jaccard_index(self,shingleA,shingleB):
          A = set(shingleA)
          B = set(shingleB)
          return ((1.0*len(A&B))/len(A|B))

      #must be DNA strings
      def determine_diss_matrix(self,mrfs,cycle_lables,verbose=False):
          diss_matrix = np.zeros((len(mrfs),len(mrfs)),dtype=float)
          for k in range(len(mrfs)):
              print(str(k))
              for j in range(k+1,len(mrfs)):
                
                
                shinglesA = self.determine_2_shingles(mrfs[k],cycle_lables[k])
           	shinglesB = self.determine_2_shingles(mrfs[j],cycle_lables[j])
           	diss_matrix[k,j] = 1.0 - self.determine_Jaccard_index(shinglesA,shinglesB) 
           	diss_matrix[j,k] = diss_matrix[k,j] 
		if verbose:
                   print(k)
                   print(mrfs[k])
                   print(len(mrfs[k])) 
                   print(cycle_lables[k])
                   print(shinglesA)
                   print(j)
                   print(mrfs[j])
                   print(len(mrfs[j]))
                   print(cycle_lables[j])
                   print(shinglesB)
                   print(diss_matrix[k,j]) 
                            
          return diss_matrix

      def generate_all_shingles(self):
          shingle_set = []
          s = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O"]
          adj_matrix = np.array([[0,1,1,0,0,0,1,0,1,1,1,0,0,0,0],
				[1,0,1,0,1,0,0,1,0,0,0,1,1,0,0],
				[1,1,0,1,0,1,0,0,0,0,0,0,0,1,1],
				[0,0,1,0,1,0,0,0,0,0,0,1,0,0,0],
				[0,1,0,1,0,0,0,0,0,0,0,0,0,1,0],
				[0,0,1,0,0,0,1,0,0,1,0,0,0,0,0],
				[1,0,0,0,0,1,0,0,0,0,0,0,0,0,1],
				[0,1,0,0,0,0,0,0,1,0,1,0,0,0,0],
				[1,0,0,0,0,0,0,1,0,0,0,0,1,0,0],
				[1,0,0,0,0,0,0,1,0,0,0,0,1,0,0],
				[1,0,0,0,0,1,0,0,0,0,0,0,0,0,1],
				[0,1,0,0,0,0,0,0,1,0,1,0,0,0,0],
				[0,1,0,1,0,0,0,0,0,0,0,0,0,1,0],
				[0,0,1,0,0,0,1,0,0,1,0,0,0,0,0],
				[0,0,1,0,1,0,0,0,0,0,0,1,0,0,0]])
          for k in range(15):
              for j in range(15):
                  if adj_matrix[k,j] == 1:
                     shingle_set.append(s[k]+s[j])

          return shingle_set
          
                 
      #must be DNA strings
      def determine_hamiltonian_cycle(self,mrfs):
          hamiltonian_cycle_label = np.zeros((len(mrfs),),dtype=int)
          s = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O"]
          adj_matrix = np.array([[0,1,1,0,0,0,1,0,1,1,1,0,0,0,0],
				[1,0,1,0,1,0,0,1,0,0,0,1,1,0,0],
				[1,1,0,1,0,1,0,0,0,0,0,0,0,1,1],
				[0,0,1,0,1,0,0,0,0,0,0,1,0,0,0],
				[0,1,0,1,0,0,0,0,0,0,0,0,0,1,0],
				[0,0,1,0,0,0,1,0,0,1,0,0,0,0,0],
				[1,0,0,0,0,1,0,0,0,0,0,0,0,0,1],
				[0,1,0,0,0,0,0,0,1,0,1,0,0,0,0],
				[1,0,0,0,0,0,0,1,0,0,0,0,1,0,0],
				[1,0,0,0,0,0,0,1,0,0,0,0,1,0,0],
				[1,0,0,0,0,1,0,0,0,0,0,0,0,0,1],
				[0,1,0,0,0,0,0,0,1,0,1,0,0,0,0],
				[0,1,0,1,0,0,0,0,0,0,0,0,0,1,0],
				[0,0,1,0,0,0,1,0,0,1,0,0,0,0,0],
				[0,0,1,0,1,0,0,0,0,0,0,1,0,0,0]])
          k = 0
          for mrf in mrfs:
              
              if adj_matrix[s.index(mrf[14]), s.index(mrf[0])] == 1:
                 hamiltonian_cycle_label[k] = 1
                 
              k+=1
          return hamiltonian_cycle_label   


      def convert_to_DNA_string(self,factors,alphabet = 2):
          if alphabet == 2:
             inner_box = ['00', '11', '010', '101']
             inner_box_str = ['A','B','C','D']
             city_str = ['Abu Dhabi', 'Bucharest', 'Chicago', 'Detroit']
          else:
             inner_box = ['00', '11', '22', '010', '020', '101', '121', '202', '212', '0120', '0210', '1021', '1201', '2012', '2102']      
             inner_box_str = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O']
             city_str = ['Abu Dhabi', 'Bucharest', 'Chicago', 'Detroit', 'Essen', 'Fresno','Gouda', 'Houston', 'Indianapolis', 'Johannesburg', 'Knoxville', 'Los Angeles', 'Moscow', 'New York', 'Oakland']

          final_string = ""
          city_string = ""
          for f in factors:
              if len(f) > 1:
                 final_string += inner_box_str[inner_box.index(f)]
                 city_string += city_str[inner_box.index(f)]+" -- " 
          
          return final_string,city_string[:-4]  

      


      def factorize(self,s="012201020112012002101210212021"):
          last_box_id = -1
          current_id = 0

          s_factors = []
 
          while (current_id < len(s)):
                s_factor, new_box_id = self.get_next_box(s,current_id)
                
                if (self.is_inner_box(s_factor)):
                   s_factors.append(s_factor)
                   current_id += 1
                   last_box_id = new_box_id
                else:
                   if (current_id > last_box_id):
                      s_factors.append(s[current_id])
                   current_id += 1

          return s_factors

      def format_output(self,s_factors):

          out_string = ""
          for k in range(len(s_factors)-1):
              o = self.get_overlap2(s1=s_factors[k],s2=s_factors[k+1])
              out_string = out_string + s_factors[k]+"--"+str(o)+"--"
          out_string = out_string + s_factors[-1]

          return out_string     

      def get_next_box(self,s="11231231",indx=0):
          indx2 = s.find(s[indx], indx+1)
          #print(indx2)
          if indx2 == -1:
             return "",indx2;
          return s[indx:indx2+1],indx2;

      def get_overlap2(self,s1="0120", s2="1201"):
          l1 = len(s1)
          l2 = len(s2)
          l = 0
          if l1 < l2:
             l = l1
          else:
             l = l2
          
          overlap = 0;
          
          for k in range(1,l):
              if s1[-1*k:] == s2[:k]:
                 overlap = k
          
          return overlap
                  
      def get_overlap(self,s1="0120", s2="1201"):
          s = difflib.SequenceMatcher(None, s1, s2)
          print("s = ",s)
          #print("s2 = ",s2)
          pos_a, pos_b, size = s.find_longest_match(0, len(s1), 0, len(s2)) 
          return s1[pos_a:pos_a+size]

      def is_inner_box(self,s="0120"):
          if len(s) == 0: 
             return False
          new_s = s[1:-1]
          if len(new_s) == 0:
             return True

          return self.is_unique(new_s)
          
      def is_unique(self,input_ar):
          char_seen = []
          for char in input_ar:
              if char in char_seen:
                 return False
              char_seen.append(char)
          return True

      def num_innerbox(self,n):
          sumv = 0
          for k in range(2,n+2):
              sumv += math.factorial(n)/math.factorial(n-k+1)

          return sumv

      def len_num_innerbox(self,n):
          sumv = 0
          for k in range(2,n+2):
              sumv += k*(math.factorial(n)/math.factorial(n-k+1))

          return sumv

      def mrf_len(self,n):
          return (n+2)*(self.num_innerbox(n)-1) - self.len_num_innerbox(n) + 2*(n+1)
    
    
      def read_file(self,name="3.txt"):
          f = open(name, "r")
          s = 0
          l = 0
          for x in f:
              #print(x)
              
              if x[0]<>'#':
                 if s == 0:
                    l = len(x[0:-1])
                 s += 1
          f.close()
          #np.array
          #print(s)
          #print(l) 
          mrfs = np.empty(s,dtype='S'+str(l))
          #print(mrfs)
          s = 0
          f = open(name, "r")
          for x in f:
              #print(x)
              
              if x[0]<>'#':
                 #print(x[0:-1])
                 mrfs[s] = x[0:-1]
                 s += 1

          return mrfs 


      def create_inner_box_strings_manual(self):
	  inner_boxes = np.empty(15,dtype='S'+str(4))
          inner_boxes[0] = "00"
          inner_boxes[1] = "11"
          inner_boxes[2] = "22"
          
          inner_boxes[3] = "010"
          inner_boxes[4] = "020"
   	  inner_boxes[5] = "101"
          inner_boxes[6] = "121"
          inner_boxes[7] = "202"
          inner_boxes[8] = "212"
          
          inner_boxes[9] = "0120"
          inner_boxes[10] = "0210"
          inner_boxes[11] = "1021"
          inner_boxes[12] = "1201"
	  inner_boxes[13] = "2012"
          inner_boxes[14] = "2102"

          return inner_boxes

      def count_boxes_n(self,s=10212,n=3):
          num_boxes = 0
          for k in range(3):
              #print("k = "+str(k))
              #print("num_boxes = "+str(num_boxes))
              if s.count(str(k)) > 0:
                 num_boxes += s.count(str(k))-1
          return num_boxes

      def create_innerbox_matrix(self):

          inner_boxes = self.create_inner_box_strings_manual()
          overlap = 0
          l = 0
          num_boxes = 0
          output_string = ""
          new_string = ""
          for k in xrange(len(inner_boxes)):
              output_string = ""
              for j in xrange(len(inner_boxes)):
                  if (len(inner_boxes[k])==2) and (len(inner_boxes[j])==2):
                     overlap = 1
                     l = 5
                     num_boxes = 2 
                  else:
                  	#print(overlap)
                        overlap =  self.get_overlap2(inner_boxes[k],inner_boxes[j])
                        new_string = inner_boxes[k]+inner_boxes[j][overlap:]
                        #print("###############")
                        #print("k = ",k)
                        #print("j= ",j)
                        #print("i[k] = ",inner_boxes[k])
                        #print("i[j] = ",inner_boxes[j])
                        #print(inner_boxes[j][overlap:])
                        #print(new_string)
                        #print("overlap = ",overlap)
                        l = len(new_string)
                        #print("l = ",l)
                        num_boxes = self.count_boxes_n(s=new_string,n=3)
                        #print("num_boxes = ",num_boxes)
                        #print("###############")
                        #print(overlap)
                        #print(new_string)
                        #break;
                        
                  	overlap = overlap*(-1) 
                  #output_string += str(l) + "(" +str(overlap)+ ")" + "["+str(num_boxes)+"];"
                  output_string += str(l*1.0/num_boxes)+";"
              print(output_string[:-1]) 

      def midway_point(self,x1,y1,x2,y2,k,p=False):
            new_x_1 = x1 + 0.5*(k)*(x2-x1)
            new_y_1 = y1 + 0.5*(k)*(y2-y1)
            new_x_2 = x2 + 0.5*(k)*(x1-x2)
            new_y_2 = y2 + 0.5*(k)*(y1-y2)
            if p:
            	plt.plot(x1,y1,"ro")
            	plt.plot(x2,y2,"go")#k % van groen af
                plt.plot(new_x_1,new_y_1,"rx")
                plt.plot(new_x_2,new_y_2,"gx")
                plt.show()
            return (new_x_1,new_y_1,new_x_2,new_y_2)

      def test_MDS(self):
          foods_binary = np.random.randint(2, size=(100, 10)) #initial dataset
          print(foods_binary.shape)
          dis_matrix = pairwise_distances(foods_binary, metric = 'jaccard')
          print(dis_matrix)
          print(dis_matrix.shape)	
          mds_model = manifold.MDS(n_components = 2, random_state = 123,
          dissimilarity = 'precomputed')
          mds_fit = mds_model.fit(dis_matrix)  
          mds_coords = mds_model.fit_transform(dis_matrix) 
                                                                                                                                  
          food_names = ['pasta', 'pizza', 'meat', 'eggs', 'cheese', 'ananas', 'pear', 'bread', 'nuts', 'milk']
          plt.figure()
          plt.scatter(mds_coords[:,0],mds_coords[:,1],facecolors = 'none', edgecolors = 'none')  # points in white (invisible)
          labels = food_names
          for label, x, y in zip(labels, mds_coords[:,0], mds_coords[:,1]):
              plt.annotate(label, (x,y), xycoords = 'data')
          plt.xlabel('First Dimension')
          plt.ylabel('Second Dimension')
          plt.title('Dissimilarity among food items')    
          plt.show()

      def eigsorted(self,cov):
          vals, vecs = np.linalg.eigh(cov)
          order = vals.argsort()[::-1]
          return vals[order], vecs[:,order]

      
      def shingle_counter(self,all_shingles,mrfs,cycle_label):

          s_counter = np.zeros((len(all_shingles),),dtype=float)
          k = 0
          for mrf in mrfs:
              print(k)
              shingles = self.determine_2_shingles(mrf,cycle_label[k])
              for s in shingles:
                  s_counter[all_shingles.index(s)] += 1
              k+=1
          return s_counter#/sum(s_counter)



      def generate_color_labels(self,all_shingles,rgb_vector,mrfs,cycle_label,s_counter):

          color_labels = np.zeros((len(mrfs),3),dtype=float)
          k = 0
          for mrf in mrfs:
              print(k)
              shingles = self.determine_2_shingles(mrf,cycle_label[k])
              rgb_value = np.zeros((3,),dtype=float)
              den = 0
              for s in shingles:
                  rgb_value = rgb_value + rgb_vector[all_shingles.index(s),:]*s_counter[all_shingles.index(s)]
                  den += s_counter[all_shingles.index(s)]
              color_labels[k,:] = rgb_value/den
              k += 1
          return color_labels

      def generate_color_labels_max(self,all_shingles,rgb_vector,mrfs,cycle_label,s_counter):

          color_labels = np.zeros((len(mrfs),3),dtype=float)
          k = 0
          for mrf in mrfs:
              print(k)
              shingles = self.determine_2_shingles(mrf,cycle_label[k])
              #rgb_value = np.zeros((3,),dtype=float)
              den = 0
              max_v = 0
              for s in shingles:
                  if s_counter[all_shingles.index(s)] > max_v:
                     den = all_shingles.index(s)
                     max_v = s_counter[all_shingles.index(s)]
              color_labels[k,:] = rgb_vector[den]
              k += 1
          return color_labels

      def generate_color_labels_min(self,all_shingles,rgb_vector,mrfs,cycle_label,s_counter):

          color_labels = np.zeros((len(mrfs),3),dtype=float)
          k = 0
          for mrf in mrfs:
              print(k)
              shingles = self.determine_2_shingles(mrf,cycle_label[k])
              #rgb_value = np.zeros((3,),dtype=float)
              den = 0
              min_v = 5000
              for s in shingles:
                  if s_counter[all_shingles.index(s)] < min_v:
                     den = all_shingles.index(s)
                     min_v = s_counter[all_shingles.index(s)]
              color_labels[k,:] = rgb_vector[den]
              k += 1
          return color_labels

      def generate_color_labels_c(self,all_shingles,rgb_vector,mrfs,cycle_label,s_counter,f=0):

          color_labels = np.zeros((len(mrfs),3),dtype=float)
          k = 0
          for mrf in mrfs:
              print(k)
              shingles = self.determine_2_shingles(mrf,cycle_label[k])
              #rgb_value = np.zeros((3,),dtype=float)
              c = []
              i = []
              for s in shingles:
                  c.append(s_counter[all_shingles.index(s)])
                  i.append(all_shingles.index(s))
              c_arr = np.array(c,dtype=int)
              i_arr = np.array(i,dtype=int)
              idx = np.argsort(c_arr)
              i_arr = i_arr[idx]
              i_arr = i_arr[::-1]
              color_labels[k,:] = rgb_vector[i_arr[f]]
              k += 1
          return color_labels

      def generate_color_labels_c_average(self,all_shingles,rgb_vector,mrfs,cycle_label,s_counter,f=0):

          color_labels = np.zeros((len(mrfs),3),dtype=float)
          k = 0
          for mrf in mrfs:
              print(k)
              shingles = self.determine_2_shingles(mrf,cycle_label[k])
              #rgb_value = np.zeros((3,),dtype=float)
              c = []
              i = []
              for s in shingles:
                  c.append(s_counter[all_shingles.index(s)])
                  i.append(all_shingles.index(s))
              c_arr = np.array(c,dtype=int)
              i_arr = np.array(i,dtype=int)
              rgb_vector_new = rgb_vector[i_arr,:]
              idx = np.argsort(c_arr)
              c_arr = c_arr[idx]
              c_arr = c_arr[::-1]
              i_arr = i_arr[idx]
              i_arr = i_arr[::-1]
              rgb_vector_new = rgb_vector_new[idx,:]
              rgb_vector_new = rgb_vector_new[::-1,:]
              color_labels[k,:] = np.sum(rgb_vector_new[0:f+1,:],axis=0)/(f+1.0)
              k += 1
          return color_labels

      def generate_color_labels_c_average_8(self,all_shingles,rgb_vector,mrfs,cycle_label):
          
          unique_labels = []

          color_labels = np.zeros((len(mrfs),3),dtype=float)
          k = 0
          for mrf in mrfs:
              print(k)
              shingles = self.determine_2_shingles(mrf,cycle_label[k])
              den = 0
              rgb_value = np.zeros((3,),dtype=float)
              item = [0,0,0,0,0,0,0,0]
              for s in shingles:
                  if s in all_shingles:
                     rgb_value += rgb_vector[all_shingles.index(s),:]
                     item[all_shingles.index(s)] = 1
                     den +=1
              if item not in unique_labels:
                 unique_labels.append(item)    
              color_labels[k,:] = rgb_value/den
              k+=1
              
          return color_labels,unique_labels

      def generate_color_labels_c_average_8(self,all_shingles,rgb_vector,mrfs,cycle_label):
          
          unique_labels = []

          color_labels = np.zeros((len(mrfs),3),dtype=float)
          k = 0
          for mrf in mrfs:
              print(k)
              shingles = self.determine_2_shingles(mrf,cycle_label[k])
              den = 0
              rgb_value = np.zeros((3,),dtype=float)
              item = [0,0,0,0,0,0,0,0]
              for s in shingles:
                  if s in all_shingles:
                     rgb_value += rgb_vector[all_shingles.index(s),:]
                     item[all_shingles.index(s)] = 1
                     den +=1
              if item not in unique_labels:
                 unique_labels.append(item)    
              color_labels[k,:] = rgb_value/den
              k+=1
              
          return color_labels,unique_labels

      def generate_color_labels_17(self,all_shingles,rgb_vector,mrfs,cycle_label,u):
          
          color_labels = np.zeros((len(mrfs),3),dtype=float)
          k = 0
          for mrf in mrfs:
              print(k)
              shingles = self.determine_2_shingles(mrf,cycle_label[k])
              den = 0
              rgb_value = np.zeros((3,),dtype=float)
              item = [0,0,0,0,0,0,0,0]
              for s in shingles:
                  if s in all_shingles:
                     item[all_shingles.index(s)] = 1
              color_labels[k,:] = rgb_vector[u.index(item),:]
              k+=1
              
          return color_labels
 
 
 
      def generate_shingle_index(self,all_shingles,mrfs,cycle_label,f=0):

          idx = np.zeros((len(mrfs),),dtype=int)
          k = 0
          s = all_shingles[f]
          for mrf in mrfs:
              shingles = self.determine_2_shingles(mrf,cycle_label[k])
              if s in shingles:
                 idx[k] = 1
              k+=1

          return idx

      def generate_region_labels(self,mds_coords):
          label_regions = np.zeros((len(mds_coords),),dtype=int)

          for k in range(len(mds_coords)):
              x = mds_coords[k,0]
              y = mds_coords[k,1]
              if y > x+0.8:
                 label_regions[k] = 1
              elif y < x-0.8: 
                 label_regions[k] = 2
              elif y > -1*x+0.35:
                 label_regions[k] = 3
              elif y < -1*x-0.35:
                 label_regions[k] = 4
              elif y > -1*x+0.17:
                 label_regions[k] = 5
              elif y < -1*x-0.15:
                 label_regions[k] = 6

          return label_regions

      def compute_shingle_densities(self,mrfs,label,all_shingles,cycle_label,r=7):
          dens = np.zeros((r,len(all_shingles)),dtype=float)

          for k in range(r):
              print(k)
              mrfs_r = mrfs[label == k]
              cycle_r = cycle_label[label==k]
              i=0
              for mrf in mrfs_r:
                  shingles = self.determine_2_shingles(mrf,cycle_r[i])
                  for s in shingles:
                      dens[k,all_shingles.index(s)] += 1
                  i+=1
              dens[k,:] = dens[k,:]/np.sum(dens[k,:])
          return dens

      def compute_shingle_regions(self,mrfs,all_shingles,cycle_label,pattern=np.array([20,23,46,52])):
          label = np.zeros((len(mrfs),),dtype=int)
          i = 0
          all_shingles = np.array(all_shingles)
          for mrf in mrfs:
              shingle_subset = all_shingles[pattern]
              shingle_subset = shingle_subset.tolist();
              shingles = self.determine_2_shingles(mrf,cycle_label[i])
              
              for s in shingle_subset:
                  if s in shingles:
                     label[i] += 1
              i+=1  
          
          return label

      def extract_specific_shingle_pattern(self,mrfs,all_shingles,cycle_label,pattern=np.array([20,23,46,52]),not_pattern=np.array([])):
          label = np.ones((len(mrfs),),dtype=int)
          i = 0
          all_shingles = np.array(all_shingles)
          for mrf in mrfs:
              shingle_subset = all_shingles[pattern]
              shingle_subset = shingle_subset.tolist();
              shingles = self.determine_2_shingles(mrf,cycle_label[i])
              
              
              for s in shingle_subset:
                  if s not in shingles:
                     label[i] = 0

              if len(not_pattern) > 0:
                 shingle_subset2 = all_shingles[not_pattern]
                 shingle_subset2 = shingle_subset2.tolist();
              
                 for s in shingle_subset2:
                     if s in shingles:
                        label[i] = 0
              
              i+=1  
          
          return label

      def compute_hamming_distance_matrix(self,u):
          d = np.zeros((len(u),len(u)),dtype=float)
          for k in range(len(u)):
              b1 = u[k]
              for i in range(k+1,len(u)):
                  t = 0;
                  b2 = u[i]
                  for j in range(len(b1)):
                      if b1[j]<>b2[j]:
                         t+=1
                  d[k,i] = t
                  d[i,k] = t
          return d

 

if __name__ == "__main__":
   b = BoxAnalysis()
   
   print("passed")
   #import kshingle as ks
   #shingles = ks.shingleseqs_list("aBc DeF", klist=[2, 5])
   #print(shingles)
   #b.test_MDS()

   #b.midway_point(-1,-1,1,1,0.5)
   #symbs = [u'\u255a', u'\u2554', u'\u2569', u'\u2566', u'\u2560', u'\u2550', u'\u256c']
   #for sym in symbs:
   #  print(sym)
   #s = "01234"

   #print(s[1:])
   #s1 = "010"
   #s2 = "101"

   #print(s1[-2:])
   #print(s2[:2])

   #o = b.get_overlap2(s1="00",s2="010")
   #print(o)
   #b.create_innerbox_matrix()

   
   generate_m = False
   generate_c = False
   
   a = "3"
   if a == "2":
      k_len = 4
   else:
      k_len = 15

   mrfs = b.read_file(name=a+".txt")
   
   #print(b.num_innerbox(2))
   #print(b.num_innerbox(3))
   #print(b.num_innerbox(4))
   #print(b.num_innerbox(5))

   #print(b.len_num_innerbox(2))
   #print(b.len_num_innerbox(3))
   #print(b.len_num_innerbox(4))
   #print(b.len_num_innerbox(5))

   #print(b.mrf_len(2))
   #print(b.mrf_len(3))
   #print(b.mrf_len(4))
   #print(b.mrf_len(5))
   #print(b.mrf_len(6))
   #print("01220121020102101202120021102")
   #print(b.format_output(b.factorize(s="01220121020102101202120021102")))

   DNA_strings = np.empty(len(mrfs),dtype='S'+str(k_len))
   
   t = 0
   for mrf in mrfs:
       print(mrf)
       f = b.factorize(s=mrf)
       s,c = b.convert_to_DNA_string(f,int(a))
       DNA_strings[t] = s
       print(b.format_output(b.factorize(s=mrf)))
       print(s)
       print(c)
       t+=1

   
   #k = 0
   #for mrf in DNA_strings:
   #    if mrf == "CDEBHIMNGOLKFJA":
   #       print(cycle_labels[k])
   #       shingles = b.determine_2_shingles(mrf,cycle_labels[k]) 
   #       print(shingles)
   #    k += 1
   
   if generate_m:
      if a == "3":
         cycle_labels = b.determine_hamiltonian_cycle(DNA_strings)
      else:
         cycle_labels = np.ones((len(DNA_strings),),dtype = int)
      #print(cycle_labels)

      diss_matrix = b.determine_diss_matrix(DNA_strings,cycle_labels)
      
      dbfile = open('dissM'+a+'.p', 'wb')
      pickle.dump(diss_matrix, dbfile)
      pickle.dump(cycle_labels, dbfile)                     
      dbfile.close()

   else:
      
      dbfile = open('dissM'+a+'.p', 'rb')     
      diss_matrix = pickle.load(dbfile)
      cycle_labels = pickle.load(dbfile)
      dbfile.close()
   
   print(diss_matrix)
   print(cycle_labels)

   if generate_c:
      mds_model = manifold.MDS(n_components = 2, random_state = 280,dissimilarity = 'precomputed',max_iter=100,n_init=1)
      mds_fit = mds_model.fit(diss_matrix)  
      mds_coords = mds_model.fit_transform(diss_matrix) 
      dbfile = open('coordMDS'+a+'.p', 'wb')
      pickle.dump(mds_coords, dbfile)                     
      dbfile.close()
   else:
      dbfile = open('coordMDS'+a+'.p', 'rb')     
      mds_coords = pickle.load(dbfile)
      dbfile.close()

   print(mds_coords)
   
   colormap = np.array(['r', 'g'])
   plt.scatter(mds_coords[:,0],mds_coords[:,1],c=colormap[cycle_labels])
   plt.gca().set_aspect('equal')
   plt.show()

   m = np.array([[0.3,0.3],[0,0],[-0.3,-0.3]])
   gm = GaussianMixture(n_components=3, random_state=200, means_init = m).fit(mds_coords)
   y_labels = gm.predict(mds_coords)
   y_prob = gm.predict_proba(mds_coords)

   plt.clf()
   ax = plt.gca()
   nstd = 2.0
   colormap = np.array(['r', 'g', 'b'])
   
   #DRAW ELLIPSE
   for i, (mean, cov) in enumerate(zip(gm.means_, gm.covariances_)):
  
   	vals, vecs = b.eigsorted(cov[:,:])
        theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
        w, h = 2 * nstd * np.sqrt(vals)
        ell1 = Ellipse(xy=(mean[0], mean[1]),width=w, height=h,angle=theta, edgecolor=colormap[i],facecolor='white',fill=True,linewidth=3,zorder=1)
        ell1.set_facecolor('none')
        ax.add_artist(ell1)

   
   ax.scatter(mds_coords[:,0],mds_coords[:,1],c=y_prob,alpha=0.2)
   plt.gca().set_aspect('equal')
   plt.show()

   phi = np.linspace(0, 2*np.pi, 8)
   rgb_cycle = np.vstack((            # Three sinusoids
    .5*(1.+np.cos(phi          )), # scaled to [0,1]
    .5*(1.+np.cos(phi+2*np.pi/3)), # 120 phase shifted.
    .5*(1.+np.cos(phi-2*np.pi/3)))).T # Shape = (54,3)
   
   #x = np.sin(phi)
   #y = np.cos(phi)
   
   #fig, ax = plt.subplots(figsize=(3,3))
   #ax.scatter(x,y, c=rgb_cycle, s=90)
   #plt.show()
   all_shingles = b.generate_all_shingles()
   idx = [19,20,22,23,46,47,52,53]
   sub_shing = [all_shingles[i] for i in idx]
   l,u = b.generate_color_labels_c_average_8(sub_shing,rgb_cycle,DNA_strings,cycle_labels)
   print(u)
   print(len(u))

   phi = np.linspace(0, 2*np.pi, len(u))
   rgb_cycle = np.vstack((            # Three sinusoids
    .5*(1.+np.cos(phi          )), # scaled to [0,1]
    .5*(1.+np.cos(phi+2*np.pi/3)), # 120 phase shifted.
    .5*(1.+np.cos(phi-2*np.pi/3)))).T # Shape = (54,3)
   
   l = b.generate_color_labels_17(sub_shing,rgb_cycle,DNA_strings,cycle_labels,u)
   
   #print(len(all_shingles))
   #col = np.array(["r","g","b","k","m"])
   #l = b.compute_shingle_regions(DNA_strings, all_shingles, cycle_labels, pattern=np.array([19,20,22,23,46,47,52,53]))
   plt.scatter(mds_coords[:,0],mds_coords[:,1],c=l,alpha=0.15) 
   plt.xlabel("X")
   plt.ylabel("Y")
   plt.gca().set_aspect('equal')
   plt.show()

   d = b.compute_hamming_distance_matrix(u)
   print(d)
   mds_model = manifold.MDS(n_components = 2, random_state = 280,dissimilarity = 'precomputed',max_iter=100,n_init=1)
   mds_fit = mds_model.fit(d)  
   mds_coords2 = mds_model.fit_transform(d) 
 
   s = np.empty((len(u),),dtype="S100")

   for k in range(len(s)):
       b_temp = u[k]
       s[k] = ""
       for i in range(len(b_temp)):
           if b_temp[i] == 1:
              s[k] += sub_shing[i]+"-"
       t = s[k]
       s[k] = t[:-1]
       
   plt.scatter(mds_coords2[:,0],mds_coords2[:,1],c=rgb_cycle,alpha=0.25,s=100) 

   for label, x, y in zip(s, mds_coords2[:,0], mds_coords2[:,1]):
       plt.annotate(label, (x-0.28,y-0.3), xycoords = 'data')
   
   plt.xlabel("X")
   plt.ylabel("Y")
   plt.gca().set_aspect('equal')
   plt.show()

   s_dens = np.zeros((6,54),dtype = float)

   l = b.extract_specific_shingle_pattern(DNA_strings,all_shingles,cycle_labels,pattern=np.array([20,23,46,52]))
   colormap = np.array(['r', 'g'])
   plt.scatter(mds_coords[:,0],mds_coords[:,1],c=colormap[l],alpha=0.25) 
   plt.xlabel("X")
   plt.ylabel("Y")
   plt.gca().set_aspect('equal')
   plt.show()

   DNA_strings_center = DNA_strings[l==1]
   cycle_labels_center = cycle_labels[l==1]
   mds_coords_center = mds_coords[l==1,:]

   #all_shingles = b.generate_all_shingles()
   print(all_shingles)
   idx = [20,23,28,29,40,41,46,52]
   sub_shing = [all_shingles[i] for i in idx]
   l,u = b.generate_color_labels_c_average_8(sub_shing,rgb_cycle,DNA_strings_center,cycle_labels_center)
   print(u)
   print(len(u))

   phi = np.linspace(0, 2*np.pi, len(u)+1)
   rgb_cycle = np.vstack((            # Three sinusoids
    .5*(1.+np.cos(phi          )), # scaled to [0,1]
    .5*(1.+np.cos(phi+2*np.pi/3)), # 120 phase shifted.
    .5*(1.+np.cos(phi-2*np.pi/3)))).T # Shape = (54,3)
   
   l = b.generate_color_labels_17(sub_shing,rgb_cycle,DNA_strings_center,cycle_labels_center,u)
   
   #print(len(all_shingles))
   #col = np.array(["r","g","b","k","m"])
   #l = b.compute_shingle_regions(DNA_strings, all_shingles, cycle_labels, pattern=np.array([19,20,22,23,46,47,52,53]))
   plt.scatter(mds_coords_center[:,0],mds_coords_center[:,1],c=l,alpha=0.5) 
   plt.xlabel("X")
   plt.ylabel("Y")
   plt.gca().set_aspect('equal')
   plt.show()

   d = b.compute_hamming_distance_matrix(u)
   print(d)
   mds_model = manifold.MDS(n_components = 2, random_state = 280,dissimilarity = 'precomputed',max_iter=100,n_init=1)
   mds_fit = mds_model.fit(d)  
   mds_coords2 = mds_model.fit_transform(d) 
 
   s = np.empty((len(u),),dtype="S100")

   for k in range(len(s)):
       b_temp = u[k]
       s[k] = ""
       for i in range(len(b_temp)):
           if b_temp[i] == 1:
              s[k] += sub_shing[i]+"-"
       t = s[k]
       s[k] = t[:-1]
       
   plt.scatter(mds_coords2[:,0],mds_coords2[:,1],c=rgb_cycle,alpha=0.5,s=100) 

   for label, x, y in zip(s, mds_coords2[:,0], mds_coords2[:,1]):
       plt.annotate(label, (x-0.1,y-0.1), xycoords = 'data')
   
   plt.xlabel("X")
   plt.ylabel("Y")
   #plt.gca().set_aspect('equal')
   plt.show()
   print(all_shingles)

   s_dens = np.zeros((6,54),dtype = float)

   #s_dens[0,:] = b.shingle_counter(all_shingles,DNA_strings_center,cycle_labels_center)
   l = b.extract_specific_shingle_pattern(DNA_strings,all_shingles,cycle_labels,pattern=np.array([41,28,20,23,46,52]))
   DNA_strings_center = DNA_strings[l==1]
   cycle_labels_center = cycle_labels[l==1]
   s_dens[0,:] = b.shingle_counter(all_shingles,DNA_strings_center,cycle_labels_center)
   l = b.extract_specific_shingle_pattern(DNA_strings,all_shingles,cycle_labels,pattern=np.array([40,29,20,23,46,52]))
   DNA_strings_center = DNA_strings[l==1]
   cycle_labels_center = cycle_labels[l==1]
   s_dens[1,:] = b.shingle_counter(all_shingles,DNA_strings_center,cycle_labels_center)
   #l = b.extract_specific_shingle_pattern(DNA_strings,all_shingles,cycle_labels,pattern=np.array([20,23,28,46,52]))
   #DNA_strings_center = DNA_strings[l==1]
   #cycle_labels_center = cycle_labels[l==1]
   #s_dens[3,:] = b.shingle_counter(all_shingles,DNA_strings_center,cycle_labels_center)
   #l = b.extract_specific_shingle_pattern(DNA_strings,all_shingles,cycle_labels,pattern=np.array([20,23,29,46,52]))
   #DNA_strings_center = DNA_strings[l==1]
   #cycle_labels_center = cycle_labels[l==1]
   #s_dens[4,:] = b.shingle_counter(all_shingles,DNA_strings_center,cycle_labels_center)
   #l = b.extract_specific_shingle_pattern(DNA_strings,all_shingles,cycle_labels,pattern=np.array([20,23,40,46,52]),not_pattern=np.array([29]))
   #DNA_strings_center = DNA_strings[l==1]
   #cycle_labels_center = cycle_labels[l==1]
   #s_dens[5,:] = b.shingle_counter(all_shingles,DNA_strings_center,cycle_labels_center)
   plt.imshow(s_dens)
   plt.show()
   

   
   
   '''
   #print(len(phi))
   s_counter = b.shingle_counter(all_shingles,DNA_strings,cycle_labels)
   
   for k in range(1):
       color_labels = b.generate_color_labels_c(all_shingles,rgb_cycle,DNA_strings,cycle_labels,s_counter,f=k)
       plt.scatter(mds_coords[:,0],mds_coords[:,1],c=color_labels,alpha=0.1)
       plt.title(str(k))
       plt.xlabel("X")
       plt.ylabel("Y")
       x = np.linspace(-0.6,0.4,10)
       x2 = np.linspace(-0.6,0.0,10)
       #y = -1*x
       #plt.plot(x,-1*x,"r")
       #plt.plot(x,-1*x-0.15,"k")
       #plt.plot(x,-1*x-0.35,"k")
       #plt.plot(x,-1*x+0.35,"k")
       #plt.plot(x,-1*x+0.15,"k")
       #plt.plot(x2,x2+0.8,"k")
       #plt.plot(x2+0.6,(x2+0.6)-0.8,"k")
       #plt.gca().set_aspect('equal')
       #plt.savefig(str(k)+".png")
       #plt.clf()
       plt.show()
   col = np.array(["r","b","g","y","m","c","k"])

   labels = b.generate_region_labels(mds_coords)
   dens = b.compute_shingle_densities(DNA_strings,labels,all_shingles,cycle_labels)
   #dens[dens==0] = 0.1
   plt.imshow(dens)
   plt.show()
   print(dens)
   x_val = np.linspace(0,len(all_shingles)+1,len(all_shingles))
   plt.plot(x_val,dens[5,:],c=col[5])
   plt.plot(x_val,dens[6,:],c=col[6])
   plt.plot(x_val,dens[2,:],c=col[2])
   plt.plot(x_val,dens[1,:],c=col[1])
   plt.show()
   plt.scatter(mds_coords[:,0],mds_coords[:,1],c=col[labels],alpha=0.1)
   plt.show()
   #for k in range(len(col)):
   #    plt.plot(x_val,dens[k,:],c=col[k])
   #plt.show()
   '''
   '''
   N = len(all_shingles)
   ind = np.arange(N)*2  # the x locations for the groups
   width = 0.27       # the width of the bars

   fig = plt.figure()
   ax = fig.add_subplot(111)

   for k in range(7):
       ax.bar(ind+width*k,dens[k,:], width, color=col[k])
   plt.show()
 
   
   plt.scatter(mds_coords[:,0],mds_coords[:,1],c=col[labels],alpha=0.1)
   plt.show()
   '''    
   #for k in range(len(all_shingles)):
   #    idx = b.generate_shingle_index(all_shingles,DNA_strings,cycle_labels,f=k)
   #    plt.scatter(mds_coords[idx==1,0],mds_coords[idx==1,1],c=rgb_cycle[k,:],alpha=0.1)       
   #    plt.title(str(k))
   #    plt.xlabel("X")
   #    plt.ylabel("Y")
   #    plt.gca().set_aspect('equal')
   #    plt.savefig(str(k)+"_sh.png")
   #    plt.clf()
       #plt.show()
   #print(DNA_strings)
   #print(cycle_labels)

   #print(DNA_strings[y_labels==1])
   #print(cycle_labels[y_labels==1])

   #n = DNA_strings[y_labels==1]
   #p = cycle_labels[y_labels==1]

   #print(n[1750])
   #print(p[1750])

   #print(n[1751])
   #print(p[1751])

   #b.determine_diss_matrix(DNA_strings[y_labels==1],cycle_labels[y_labels==1],verbose=True)

   '''
   #k = 0 
   #for mrf in DNA_strings:
   #    sh = b.determine_2_shingles(mrf,cycle_label[k])
   #    k+=1
   #    print(mrf)
   #    print(sh)
     
   '''
   #X = np.random.uniform(0,1,(len(mrfs),2))
   #print(X)

   #for k in range(X.shape[0]):
   #    if k%10 == 0:
   #       plt.plot(X[k,0],X[k,1],"rx")
       #if f[-3] == "00":
       #   if len(f[-2])==1:
       #      print(mrf)
       #      print(b.format_output(b.factorize(s=mrf)))
   #plt.show()
   
   '''
   diss_matrix = np.zeros((len(mrfs),len(mrfs)),dtype=float)
   for k in range(len(mrfs)):
       print(str(k))
       for j in range(k+1,len(mrfs)):
           lcs_s = b.lcs(DNA_strings[k],DNA_strings[j])
           diss_matrix[k,j] = 1.0 - (1.0*len(lcs_s))/(1.0*k_len) 
           diss_matrix[j,k] = diss_matrix[k,j] 
           #if len(lcs_s) > 1:
           #   X[k,0],X[k,1],X[j,0],X[j,1] = b.midway_point(X[k,0],X[k,1],X[j,0],X[j,1],1.0*len(lcs_s)/(1.0*k_len))

   dbfile = open('dissMPickle2.p', 'wb')
   pickle.dump(diss_matrix, dbfile)                     
   dbfile.close()
   '''
   '''
   dbfile = open('dissMPickle.p', 'rb')     
   diss_matrix = pickle.load(dbfile)
   dbfile.close()
   print(diss_matrix)
   #print(db)
   mds_model = manifold.MDS(n_components = 2, random_state = 123,dissimilarity = 'precomputed',max_iter=100,n_init=1)
   mds_fit = mds_model.fit(diss_matrix)  
   mds_coords = mds_model.fit_transform(diss_matrix) 
   print(mds_coords)

   dbfile = open('mds_coords.p', 'wb')
   pickle.dump(mds_coords, dbfile)                     
   dbfile.close()
   '''
   '''
   dbfile = open('mds_coords.p', 'rb')
   mds_coords = pickle.load(dbfile)                     
   dbfile.close()
   
   r = np.amax(np.sqrt(mds_coords[:,0]**2+mds_coords[:,1]**2))
   print(r)
   th = np.arctan2(mds_coords[:,1],mds_coords[:,0])* 180 / np.pi
   print(th)
   plt.plot(th[np.logical_and(th > 80,th < 10)])
   plt.show()
   #print(DNA_strings[np.logical_and(th > 0,th < 10)])
   DNA_s = DNA_strings[np.logical_and(np.logical_and(th > 80,th < 90),r>0.3)]
   DNA_s = np.sort(DNA_s)
   print(DNA_s)
   
   list1 = []
   for k in range(len(DNA_s)):
       #print(str(k))
       for j in range(k+1,len(DNA_s)):
           lcs_s = b.lcs(DNA_s[k],DNA_s[j])
           #print(lcs_s)
           list1.append(lcs_s)
   #list1 = list(set(list1))
   list1.sort()
   print(list1)
   colormap = np.array(['r', 'g'])
   plt.scatter(mds_coords[:,0],mds_coords[:,1],c=colormap[cycle_label])
   #plt.show()
   theta = np.linspace(0, 2*np.pi, 100)

   # compute x1 and x2
   x1 = r*np.cos(theta)
   x2 = r*np.sin(theta)

   plt.plot(x1,x2,"r")
   k = 0
   for label, x, y in zip(DNA_strings, mds_coords[:,0], mds_coords[:,1]):
       if k%100 == 0:       
          plt.annotate(label, (x-0.01,y-0.01), xycoords = 'data')
       k += 1
   plt.gca().set_aspect('equal')
   plt.show()
   '''          

  
   '''
   for k in range(X.shape[0]):
       if k%10 == 0:
          plt.plot(X[k,0],X[k,1],"bx")
       #if f[-3] == "00":
       #   if len(f[-2])==1:
       #      print(mrf)
       #      print(b.format_output(b.factorize(s=mrf)))
   plt.show()

   

      
   #print(DNA_strings)
   #print(b.lcs(DNA_strings[0],DNA_strings[1]))
   #print(b.lcs(DNA_strings[0],DNA_strings[2]))
   #print(b.lcs(DNA_strings[0],DNA_strings[3]))
   #print(b.lcs(DNA_strings[0],DNA_strings[4]))
   #print(b.lcs(DNA_strings[0],DNA_strings[5]))
   #print(b.lcs(DNA_strings[0],DNA_strings[6]))
   #print(b.lcs(DNA_strings[0],DNA_strings[7]))
   #print(len(mrfs[0]))
   #print(b.get_overlap(s1="0120",s2="202"))    
   #print(b.get_overlap(s1="22",s2="2012"))  
   #print(b.get_overlap(s1="1021",s2="0210"))  
   #print(b.get_overlap(s1="123",s2="456"))  
   #print(b.get_overlap(s1="1201",s2="11"))
   #print(b.is_inner_box(s="1001"))
   #print(b.is_inner_box(s="11"))
   #print(b.is_inner_box(s="1201"))
   #print(b.get_next_box(s="13121231",indx=1))   
   #print(b.format_output(b.factorize()))
   #f.plot_mat(name="Gauteng_nochange.mat")
   '''
