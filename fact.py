import scipy.io
import pylab as plt
import math
import numpy as np
import difflib

class BoxAnalysis():

      def __init__(self):
          pass

      #def calculate_overlapping(s1,s2):
      #    f2 = s2[0]
      
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
 

          

  
if __name__ == "__main__":
   b = BoxAnalysis()
   symbs = [u'\u255a', u'\u2554', u'\u2569', u'\u2566', u'\u2560', u'\u2550', u'\u256c']
   for sym in symbs:
     print(sym)
   #s = "01234"

   #print(s[1:])
   #s1 = "010"
   #s2 = "101"

   #print(s1[-2:])
   #print(s2[:2])

   #o = b.get_overlap2(s1="00",s2="010")
   #print(o)
   #b.create_innerbox_matrix()

   #mrfs = b.read_file(name="3.txt")

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

   #for mrf in mrfs:
   #    print(mrf)
   #    #print(b.factorize(s=mrf))
   #    print(b.format_output(b.factorize(s=mrf)))   
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
