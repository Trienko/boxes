import scipy.io
import pylab as plt
import math
import numpy as np
import difflib
from termcolor import colored

import codecs, sys


class BoxAnalysis():

      def __init__(self):
          pass

      #def calculate_overlapping(s1,s2):
      #    f2 = s2[0]


      def containZeroOuterBox(self,s='10101'):
          l = [idx for idx, item in enumerate(s.lower()) if '0' in item]
          #print(l)
          return l

      def stickBackTogether(self,f):
          '''
          print(f)
          s_factors_temp = []
          s_temp = ""

          d = {}

          for i in range(len(f)-1):
              #is a innerbox
              if (f[i])[0] == '0':
                 if len(f[i]) > 0:
                    continue 
              if (f[i+1])[0] == '0':
                 if len(f[i+1]) > 0:
                    continue 
              o  = self.get_overlap2(f[i],f[i+1])
              if (len(s_temp)==0):
                 s_temp = s_temp + f[i] + (f[i+1])[o:]
              else:
                 s_temp = s_temp + (f[i+1])[o:]
              print(s_temp)
              #if len(f[i]) > 1:
              s_factors_temp.append(f[i])
              
              l = self.containZeroOuterBox(s_temp)
              if (len(l) > 1):
                 #if len(f[i+1]) > 1:  
                 s_factors_temp.append(f[i+1])
                 outer_box = s_temp[l[0]:l[1]+1]
                 d[outer_box] = s_factors_temp
                 s_temp = ""
                 s_factors_temp = []
                 #break
            '''
          new_string = True
          s_temp = ""
          s_factor = []
          d = {} 
          outer_box = ""
          l_while = len(f)
          k = 0
          print(l_while)
          #for k in range(len(f)):  
          while k < l_while:
              #print("**************")
              #k = k + 1
              #print(k)
              #print(l_while)
              #print(s_temp)
              #print(s_factor)
              #print(new_string)
              #print(outer_box)
             
              #print("**************")
              if len(self.containZeroOuterBox(f[k])) > 1: #0-innerbox?

                 if new_string: #new 0 substring?
                    k = k + 1
                    continue
                 else:
                    o = self.get_overlap2(f[k-1],f[k]) 
                    s_temp += f[k][o:]
                    l = self.containZeroOuterBox(s_temp)
                    outer_box = s_temp[l[0]:l[1]+1]
                    if len(f[k]) > 1:
                       s_factor.append(f[k])
                    if not self.isUniqueChars(outer_box[1:-1]): 
                       d[outer_box] = s_factor
                    new_string = True
                    s_temp = ""
                    s_factor = []
                    k = k - 1
              else:
                    if new_string:
                       new_string  = False
                       if len(self.containZeroOuterBox(f[k])) > 0:
                          #print("Hallo1")
                          s_temp = f[k]
                          if len(f[k]) > 1:
                             s_factor.append(f[k])
                       else:
                          #print("Hallo2")
                          s_temp = "0"+f[k]
                          if len(f[k-1]) > 1:
                             s_factor.append(f[k-1])
                          if len(f[k]) > 1:
                             s_factor.append(f[k])

                    else:
                        o = self.get_overlap2(f[k-1],f[k]) 
                        s_temp += f[k][o:]
                        l = self.containZeroOuterBox(s_temp)
                        if len(f[k]) > 1:
                             s_factor.append(f[k])
                        if len(l) > 1:
                           outer_box = s_temp[l[0]:l[1]+1]
                           if not self.isUniqueChars(outer_box[1:-1]): 
                              d[outer_box] = s_factor
                           new_string = True
                           s_temp = ""
                           s_factor = []
                           k = k - 1
              k = k + 1
              #print(k)
              #break

          return d             
              

      def isUniqueChars(self,st): 
  
          # String length cannot be more than 
          # 256. 
          if len(st) > 256: 
             return False
  
          # Initialize occurrences of all characters 
          char_set = [False] * 128
  
          # For every character, check if it exists 
          # in char_set 
          for i in range(0, len(st)): 
  
              # Find ASCII value and check if it 
              # exists in set. 
              val = ord(st[i]) 
              if char_set[val]: 
                 return False
  
              char_set[val] = True
  
          return True


      def zero_box_factor_5(self,s=""):
          s1= "1234001234103214301234021430241032401234201430213420132410342012431042134012430214320134021340231042301243210342130241320143210423140324102340123042130412031"
          s2 = "402130423104320142310241304210324130214032104321024301241301423014320412340132401320413024310234102431203421043120413240314230413204310243412043214031204312401342103412034102314"
          s3 = "023412013410214312023142013420312430142103142034130231432031424031240310240130420314123014203132042132402314402312040321402311240321304312340312103401204302103431020413402431420"
          s4 = "4131024230132431034230104234101432401423241043230412314013210412304102301403204142032142302412030124034124304214033042123404132203412303412"
          s = s1+s2+s3+s4

          #print(outer_zero_box_letter)
          #outer_zero_box_letter = ""
          #for k in range(len(o)):
          #    outer_zero_box_letter+=o[k]
          #print(outer_zero_box_letter)
          print s
          c = '0'
          x = [pos for pos, char in enumerate(s) if char == c]
          k_prev = 1
          s_factors = []
          s_blue = []
          outer = []
          for k in x:
              if k <> 0:
                 s_factors.append(s[k_prev:k+1])
              k_prev = k
          
          suffix = False 
          if s[k_prev:] <> '0':
             s_factors.append(s[k_prev:])
             suffix = True
          #print(s_factors)
          
          s_temp = s_factors[0]
          
          o_string = ""
          final_string = ""


          if self.isUniqueChars(s_temp[1:-1]):
             text = colored(s_factors[0],"red")
             #final_string = inner_zero_box_letter[inner_zero_box.index(s_factors[0])]  
          else:
             s_blue.append(s_factors[0])
             text = colored(s_factors[0],"blue")
             
             #print(np.where(outer_zero_box==s_factors[0]))
             #print(outer_zero_box)
             #print(s_factors[0])
             #outer.append(outer_zero_box.index(s_factors[0]))
             #o_string = outer_zero_box_letter[outer_zero_box.index(s_factors[0])]
             #final_string = outer_zero_box_letter[outer_zero_box.index(s_factors[0])]  
  

          for k in range(1,len(s_factors)-1):
              s_temp = s_factors[k] 
              if self.isUniqueChars(s_temp[1:-1]):
                 text = text +"--"+ colored(s_factors[k],"red")
                 #final_string += inner_zero_box_letter[inner_zero_box.index(s_factors[k])]
              else:
                 text = text + "--"+ colored(s_factors[k],"blue")
                 s_blue.append(s_factors[k])
                 #outer.append(outer_zero_box.index(s_factors[k]))
                 #o_string += outer_zero_box_letter[outer_zero_box.index(s_factors[k])]  
                 #final_string += outer_zero_box_letter[outer_zero_box.index(s_factors[k])]

          if suffix:
             text = text + "--" + colored(s_factors[-1],"green")   
          else:
              s_temp = s_factors[-1]
              if self.isUniqueChars(s_temp[1:-1]):
                 text = text +"--"+ colored(s_factors[-1],"red")
                 #final_string += inner_zero_box_letter[inner_zero_box.index(s_factors[-1])] 
              else:
                 text = text + "--"+ colored(s_factors[-1],"blue")
                 s_blue.append(s_factors[-1])
                 #o_string += outer_zero_box_letter[outer_zero_box.index(s_factors[-1])]
                 #outer.append(outer_zero_box.index(s_factors[-1]))
                 #final_string += outer_zero_box_letter[outer_zero_box.index(s_factors[-1])]
          #print(outer)
          #outer = sorted(outer)
          #print(outer)  
          #o_string = ""
          #for k in range(len(outer)):
          #    o_string = o_string + "("+str(outer[k])+")"

          print text#+"---"+o_string
          return s_blue#,o_string#,"".join(sorted(o_string)),final_string
	

      def zero_box_factor_4(self,s="01230212031230132102301130213203132021310213031213012030210312032100213321012310231032301032013201232103213023120130203102230123"):
          outer_zero_box = ['031230', '0123210', '012310', '021320', '02133210', '021310', '02120', '03230', '02230', '01130', '032130', '013210', '0312130', '023120', '031320', '012320', '01233210', '0322130', '013230', '023210', '022130', '0213320', '01322130', '022310', '0123320', '031220', '013320', '032310', '021230', '01310', '01120', '0231320', '013220', '0321230', '03220', '0132310', '01220', '0110', '031210', '032230', '01210', '013120', '012330', '011320', '023130', '0133210', '0322310', '021330', '0220', '0133120', '02311320', '0231130', '012230', '01330', '03211320', '03310', '0132230', '0233120', '03110', '0311230', '031130', '0311320', '02210', '033120', '011230', '0231120', '033210', '02311230', '023110', '0132210', '012130', '0330', '032110', '032210', '03320', '01322310', '013310', '032120', '0211230', '012210', '031120', '03211230', '0321130', '0123310', '0213120', '0321120', '023310', '0122130', '021130', '021120', '02110', '0233210', '0312230', '0211320', '02320', '0122310', '03122310', '01233120', '03130', '02330', '0312210', '03122130', '023320', '0213310']
          
          outer_zero_box_letter=[unichr(code) for code in range(32,len(outer_zero_box)+32)]    
          #print(outer_zero_box_letter)
          #outer_zero_box_letter = ""
          #for k in range(len(o)):
          #    outer_zero_box_letter+=o[k]
          #print(outer_zero_box_letter)
          print s
          c = '0'
          x = [pos for pos, char in enumerate(s) if char == c]
          k_prev = 1
          s_factors = []
          s_blue = []
          outer = []
          for k in x:
              if k <> 0:
                 s_factors.append(s[k_prev:k+1])
              k_prev = k
          
          suffix = False 
          if s[k_prev:] <> '0':
             s_factors.append(s[k_prev:])
             suffix = True
          #print(s_factors)
          
          s_temp = s_factors[0]
          
          o_string = ""
          final_string = ""


          if self.isUniqueChars(s_temp[1:-1]):
             text = colored(s_factors[0],"red")
             #final_string = inner_zero_box_letter[inner_zero_box.index(s_factors[0])]  
          else:
             s_blue.append(s_factors[0])
             text = colored(s_factors[0],"blue")
             
             #print(np.where(outer_zero_box==s_factors[0]))
             #print(outer_zero_box)
             #print(s_factors[0])
             outer.append(outer_zero_box.index(s_factors[0]))
             #o_string = outer_zero_box_letter[outer_zero_box.index(s_factors[0])]
             #final_string = outer_zero_box_letter[outer_zero_box.index(s_factors[0])]  
  

          for k in range(1,len(s_factors)-1):
              s_temp = s_factors[k] 
              if self.isUniqueChars(s_temp[1:-1]):
                 text = text +"--"+ colored(s_factors[k],"red")
                 #final_string += inner_zero_box_letter[inner_zero_box.index(s_factors[k])]
              else:
                 text = text + "--"+ colored(s_factors[k],"blue")
                 s_blue.append(s_factors[k])
                 outer.append(outer_zero_box.index(s_factors[k]))
                 #o_string += outer_zero_box_letter[outer_zero_box.index(s_factors[k])]  
                 #final_string += outer_zero_box_letter[outer_zero_box.index(s_factors[k])]

          if suffix:
             text = text + "--" + colored(s_factors[-1],"green")   
          else:
              s_temp = s_factors[-1]
              if self.isUniqueChars(s_temp[1:-1]):
                 text = text +"--"+ colored(s_factors[-1],"red")
                 #final_string += inner_zero_box_letter[inner_zero_box.index(s_factors[-1])] 
              else:
                 text = text + "--"+ colored(s_factors[-1],"blue")
                 s_blue.append(s_factors[-1])
                 #o_string += outer_zero_box_letter[outer_zero_box.index(s_factors[-1])]
                 outer.append(outer_zero_box.index(s_factors[-1]))
                 #final_string += outer_zero_box_letter[outer_zero_box.index(s_factors[-1])]
          #print(outer)
          outer = sorted(outer)
          print(outer)  
          o_string = ""
          for k in range(len(outer)):
              o_string = o_string + "("+str(outer[k])+")"

          print text+"---"+o_string
          return s_blue,o_string#,"".join(sorted(o_string)),final_string
	


      def zero_box_factor(self,s="0122010201120120021012102120210"):
          outer_zero_box = ['02120', '01120', '01210', '01220', '012210', '02110', '021120', '02210']
          outer_zero_box_letter = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']

	  inner_zero_box = ['00','010','020','0120','0210']
          inner_zero_box_letter = ['a','b','c','d','e']

          	

          print s
          c = '0'
          x = [pos for pos, char in enumerate(s) if char == c]
          k_prev = 1
          s_factors = []
          s_blue = []
          for k in x:
              if k <> 0:
                 s_factors.append(s[k_prev:k+1])
              k_prev = k
          
          suffix = False 
          if s[k_prev:] <> '0':
             s_factors.append(s[k_prev:])
             suffix = True
          s_temp = s_factors[0]
          o_string = ""
          final_string = ""

          if self.isUniqueChars(s_temp[1:-1]):
             text = colored(s_factors[0],"red")
             final_string = inner_zero_box_letter[inner_zero_box.index(s_factors[0])]  
          else:
             s_blue.append(s_factors[0])
             text = colored(s_factors[0],"blue")
             
             #print(np.where(outer_zero_box==s_factors[0]))
             #print(outer_zero_box)
             #print(s_factors[0])
             o_string = outer_zero_box_letter[outer_zero_box.index(s_factors[0])]
             final_string = outer_zero_box_letter[outer_zero_box.index(s_factors[0])]  
  

          for k in range(1,len(s_factors)-1):
              s_temp = s_factors[k] 
              if self.isUniqueChars(s_temp[1:-1]):
                 text = text +"--"+ colored(s_factors[k],"red")
                 final_string += inner_zero_box_letter[inner_zero_box.index(s_factors[k])]
              else:
                 text = text + "--"+ colored(s_factors[k],"blue")
                 s_blue.append(s_factors[k])
                 o_string += outer_zero_box_letter[outer_zero_box.index(s_factors[k])]  
                 final_string += outer_zero_box_letter[outer_zero_box.index(s_factors[k])]

          if suffix:
             text = text + "--" + colored(s_factors[-1],"green")   
          else:
              s_temp = s_factors[-1]
              if self.isUniqueChars(s_temp[1:-1]):
                 text = text +"--"+ colored(s_factors[-1],"red")
                 final_string += inner_zero_box_letter[inner_zero_box.index(s_factors[-1])] 
              else:
                 text = text + "--"+ colored(s_factors[-1],"blue")
                 s_blue.append(s_factors[-1])
                 o_string += outer_zero_box_letter[outer_zero_box.index(s_factors[-1])]
                 final_string += outer_zero_box_letter[outer_zero_box.index(s_factors[-1])]

          print text+"---"+"".join(sorted(o_string))
          return s_blue,"".join(sorted(o_string)),final_string

          #if self.isUniqueChars(s_factors[0][1:-1])
          #   text = colored(s_factors[0],"red")
          #   for k in range(1,len(s_factors)):
              

          #print s_factors
          #text = colored("hallo",'red')
          #text2 = colored("hallo",'yellow')
          #print(text+"-"+text2)
      
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
              o = len(self.get_overlap(s1=s_factors[k],s2=s_factors[k+1]))
              out_string = out_string + s_factors[k]+"--"+str(o)+"--"
          out_string = out_string + s_factors[-1]

          return out_string     

      def get_next_box(self,s="11231231",indx=0):
          indx2 = s.find(s[indx], indx+1)
          #print(indx2)
          if indx2 == -1:
             return "",indx2;
          return s[indx:indx2+1],indx2;
          
      def get_overlap(self,s1="0120", s2="1201"):
          s = difflib.SequenceMatcher(None, s1, s2)
          pos_a, pos_b, size = s.find_longest_match(0, len(s1), 0, len(s2)) 
          return s1[pos_a:pos_a+size]

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
     
      def create_adjacency_matrix(self,s="abcdeABCD",mrf=""):
          adj_matrix = np.zeros((len(s),len(s)),dtype=int)

          for k in range(len(mrf)-1):
              row = s.index(mrf[k])
              column = s.index(mrf[k+1])
              adj_matrix[row,column] = 1.0

          return adj_matrix

          
    
    
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

      def num_innerbox(self,n):
          sumv = 0
          for k in range(2,n+2):
              sumv += math.factorial(n)/math.factorial(n-k+1)

          return sumv

      def num_innerbox_0(self,n):
          sumv = 0
          for k in range(2,n+2):
              sumv += math.factorial(n-1)/math.factorial(n-k+1)

          return sumv

      def num_outerbox_0(self,n):
          sumv = 0
          for k in range(2,n+1):
              sumv += math.factorial(n-1)/math.factorial(n-k)

          return sumv


      def create_adjacency_matrix_innerboxes(self,mrf=[]):
          s = ["00","11","22","010","020","101","121","202","212","0120","0210","1021","1201","2012","2102"]
          adj_matrix = np.zeros((len(s),len(s)),dtype=int)

          mrf_new = []
          for k in range(len(mrf)):
              if len(mrf[k])>1:
                 mrf_new.append(mrf[k]) 


          for k in range(len(mrf_new)-1):
                row = s.index(mrf_new[k])
                column = s.index(mrf_new[k+1])
                adj_matrix[row,column] = 1.0
              
          return adj_matrix

      def m_div(self,m1,m2):
          m = np.zeros(m1.shape,dtype=float)
          for k in range(m1.shape[0]):
              for j in range(m1.shape[1]):
                  if m2[k,j] <> 0: 
                     m[k,j] = (m1[k,j]*1.0)/m2[k,j]
          return m  
   
      def create_graph(self,adj_m=[],name="All.png",c='b'):
          import networkx as nx  
          G=nx.DiGraph()
          G.add_node(0),G.add_node(1),G.add_node(2),G.add_node(3),G.add_node(4),G.add_node(5),G.add_node(6),G.add_node(7),G.add_node(8),G.add_node(9),G.add_node(10),G.add_node(11),G.add_node(12),G.add_node(13),G.add_node(14)
          for row in range(15):
              for column in range(15):
                  if adj_m[row,column] > 0:
                     G.add_edge(row,column,weight=adj_m[row,column])                     
          #nx.draw(G, pos=nx.circular_layout(G), with_labels=True, font_weight='bold')
          #labels={}
          #labels[0]='00'
          #labels[1]='11'
          #nx.draw_networkx_labels(G,pos=nx.circular_layout(G),labels,font_size=16)
          #plt.show()

          pos = nx.circular_layout(G)
          #pos = nx.planar_layout(G)

          # nodes
          nx.draw_networkx_nodes(G,pos,
                       nodelist=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14],
                       node_color=c,
                       node_size=600,
                   alpha=0.9)
          #nx.draw_networkx_nodes(G,pos,
          #             nodelist=[4,5,6,7],
          #             node_color='b',
          #             node_size=500,
          #         alpha=0.8)

          # edges
          nx.draw_networkx_edges(G,pos,width=1.0,alpha=0.5,with_labels=True)
          #nx.draw_networkx_edges(G,pos,
          #             edgelist=[(0,1),(1,2),(2,3),(3,0)],
          #             width=8,alpha=0.5,edge_color='r')
          #nx.draw_networkx_edges(G,pos,
          #             edgelist=[(4,5),(5,6),(6,7),(7,4)],
          #             width=8,alpha=0.5,edge_color='b')


          # some math labels
          labels={}
          labels[0]='00'
          labels[1]='11'
          labels[2]='22'
          labels[3]='010'
          labels[4]='020'
          labels[5]='101'
          labels[6]='121'
          labels[7]='202'
          labels[8]='212'
          labels[9]='0120'
          labels[10]='0210'
          labels[11]='1021'
          labels[12]='1201'
          labels[13]='2012'
          labels[14]='2102'
          
          nx.draw_networkx_labels(G,pos,labels,font_size=10)
          edge_labels=dict([((u,v,),d['weight']) 
		for u,v,d in G.edges(data=True)])
          nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, label_pos=0.3, font_size=7)
          #labels = nx.get_edge_attributes(G,'weight')
          #nx.draw_networkx_edge_labels(G,pos,edge_labels=labels)

          plt.axis('off')
          plt.savefig(name) # save as png
          plt.show() # display
     
      def create_graph2(self,adj_m=[],name="All.png",c='b'):
          import networkx as nx  
          G=nx.DiGraph()
          G.add_node(0),G.add_node(1),G.add_node(2),G.add_node(3),G.add_node(4),G.add_node(5),G.add_node(6),G.add_node(7),G.add_node(8),G.add_node(9),G.add_node(10),G.add_node(11),G.add_node(12),G.add_node(13),G.add_node(14)
          for row in range(15):
              for column in range(15):
                  if adj_m[row,column] > 0:
                     G.add_edge(row,column,weight=adj_m[row,column])                     
          #nx.draw(G, pos=nx.circular_layout(G), with_labels=True, font_weight='bold')
          #labels={}
          #labels[0]='00'
          #labels[1]='11'
          #nx.draw_networkx_labels(G,pos=nx.circular_layout(G),labels,font_size=16)
          #plt.show()

          pos = nx.circular_layout(G)
          #pos = nx.planar_layout(G)

          # nodes
          nx.draw_networkx_nodes(G,pos,
                       nodelist=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14],
                       node_color=c,
                       node_size=600,
                   alpha=0.9)
          #nx.draw_networkx_nodes(G,pos,
          #             nodelist=[4,5,6,7],
          #             node_color='b',
          #             node_size=500,
          #         alpha=0.8)

          # edges
          nx.draw_networkx_edges(G,pos,width=1.0,alpha=0.5,with_labels=True)
          nx.draw_networkx_edges(G,pos,
                       edgelist=[(11,8),(8,12),(8,0),(0,8),(8,7),(7,8)],
                       width=8,alpha=0.2,edge_color='r',)
          nx.draw_networkx_edges(G,pos,
                       edgelist=[(4,1),(1,12),(12,1),(1,0),(1,7)],
                       width=8,alpha=0.2,edge_color='g',)
          nx.draw_networkx_edges(G,pos,
                       edgelist=[(13,6),(6,14),(6,0),(0,6),(5,6),(6,5)],
                       width=8,alpha=0.2,edge_color='b',)
          nx.draw_networkx_edges(G,pos,
                       edgelist=[(13,2),(2,13),(2,3),(0,2),(5,2)],
                       width=8,alpha=0.2,edge_color='y',)
          
          
          #nx.draw_networkx_edges(G,pos,
          #             edgelist=[(4,5),(5,6),(6,7),(7,4)],
          #             width=8,alpha=0.5,edge_color='b')


          # some math labels
          labels={}
          labels[0]='00'
          labels[1]='11'
          labels[2]='22'
          labels[3]='010'
          labels[4]='020'
          labels[5]='101'
          labels[6]='121'
          labels[7]='202'
          labels[8]='212'
          labels[9]='0120'
          labels[10]='0210'
          labels[11]='1021'
          labels[12]='1201'
          labels[13]='2012'
          labels[14]='2102'
          
          nx.draw_networkx_labels(G,pos,labels,font_size=10)
          #edge_labels=dict([((u,v,),d['weight']) 
		#for u,v,d in G.edges(data=True)])
          #nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, label_pos=0.3, font_size=7)
          #labels = nx.get_edge_attributes(G,'weight')
          #nx.draw_networkx_edge_labels(G,pos,edge_labels=labels)

          plt.axis('off')
          plt.savefig(name) # save as png
          plt.show() # display
  
   
if __name__ == "__main__":

   b = BoxAnalysis()
   #b.containZeroOuterBox()
   #mrf = "012021201020121012201120021021"
   #f = b.factorize(s=mrf)
   #print(b.format_output(f))
   #print(b.zero_box_factor(mrf))
   #print(b.stickBackTogether(f))
   
   mrfs = b.read_file(name="3.txt")
   d = {}
   d["02120"] = []
   d["01210"] = []
   d["01120"] = []
   d["01220"] = []
   d["021120"] = []
   d["012210"] = []
   d["02110"] = []
   d["02210"] = []
   
   for mrf in mrfs:
       s = b.factorize(s=mrf)
       print("******************")
       print(mrf)
       print(b.zero_box_factor(mrf))
       print(b.format_output(s))
       temp = b.stickBackTogether(s)
       print(temp)
       for key in temp.keys():
           if temp[key] not in d[key]:
              d[key].append(temp[key])
       #print(b.stickBackTogether(s))
       
       print("******************")
   
   for key in d.keys():
       print("******************")  
       print(key)
       print(d[key])
       print("******************")
   #print(d)
   
   
   mrfs = b.read_file(name="3.txt")

   adj_m = np.zeros((15,15),dtype=int)
   adj_m_group_1 = np.zeros((15,15),dtype=int)
   adj_m_group_2 = np.zeros((15,15),dtype=int)
   adj_m_group_3 = np.zeros((15,15),dtype=int)
   adj_m_group_4 = np.zeros((15,15),dtype=int)
   adj_m_group_5 = np.zeros((15,15),dtype=int)
   for mrf in mrfs:
       print(mrf)
       #innerbox_factorization
       adj_m += b.create_adjacency_matrix_innerboxes(mrf=b.factorize(s=mrf))
       
       #zero_box_facotrization
       print(mrf)
       second_list,chars,final_string = b.zero_box_factor(s=mrf)
       if "".join(sorted(final_string)) == "ABCDabcde":
          #if it belongs to a certain group trace path in graph
          adj_m_group_1 += b.create_adjacency_matrix_innerboxes(mrf=b.factorize(s=mrf))
       if "".join(sorted(final_string)) == "ACDGabcde":
          #if it belongs to a certain group trace path in graph
          adj_m_group_2 += b.create_adjacency_matrix_innerboxes(mrf=b.factorize(s=mrf))
       if "".join(sorted(final_string)) == "ACEGabcde":
          #if it belongs to a certain group trace path in graph
          adj_m_group_3 += b.create_adjacency_matrix_innerboxes(mrf=b.factorize(s=mrf))
       if "".join(sorted(final_string)) == "ACFHabcde":
          #if it belongs to a certain group trace path in graph
          adj_m_group_4 += b.create_adjacency_matrix_innerboxes(mrf=b.factorize(s=mrf))
       if "".join(sorted(final_string)) == "ACEFabcde":
          #if it belongs to a certain group trace path in graph
          adj_m_group_5 += b.create_adjacency_matrix_innerboxes(mrf=b.factorize(s=mrf))
       
       
       
       
          #factored_group1.append(final_string)

       #print(b.factorize(s=mrf))
       #print(b.format_output(b.factorize(s=mrf)))   
   #print(adj_m)
   #print(adj_m_group_1)
   #b.create_graph(adj_m=adj_m,name="All.png",c='b')
   #b.create_graph(adj_m=(100*b.m_div(adj_m_group_1,adj_m)).astype(int),name="Group1.png",c='r')

   b.create_graph2(adj_m=(100*b.m_div(adj_m_group_1,adj_m)).astype(int),name="Group1.png",c='r')

   #b.create_graph(adj_m=(100*b.m_div(adj_m_group_2,adj_m)).astype(int),name="Group2.png",c='g')
   #b.create_graph(adj_m=(100*b.m_div(adj_m_group_3,adj_m)).astype(int),name="Group3.png",c='y')
   #b.create_graph(adj_m=(100*b.m_div(adj_m_group_4,adj_m)).astype(int),name="Group4.png",c='c')
   #b.create_graph(adj_m=(100*b.m_div(adj_m_group_5,adj_m)).astype(int),name="Group5.png",c='m')

   plt.imshow(b.m_div(adj_m_group_1,adj_m),vmin=0,vmax=1)
   plt.show()
   plt.imshow(b.m_div(adj_m_group_2,adj_m),vmin=0,vmax=1)
   plt.show()
   plt.imshow(b.m_div(adj_m_group_3,adj_m),vmin=0,vmax=1)
   plt.show()
   plt.imshow(b.m_div(adj_m_group_4,adj_m),vmin=0,vmax=1)
   plt.show()
   plt.imshow(b.m_div(adj_m_group_5,adj_m),vmin=0,vmax=1)
   plt.show()
   
   '''
   b.zero_box_factor()
   b_factors = []
   chars_c = []
   mrfs = b.read_file(name="3.txt")
   factored_abc = []
   factored_group1 = []
   for mrf in mrfs:
       second_list,chars,final_string = b.zero_box_factor(s=mrf)
       b_factors = b_factors + list(set(second_list) - set(b_factors))
       factored_abc.append(final_string)
       if not chars in chars_c:
          chars_c.append(chars)
       if "".join(sorted(final_string)) == "ABCDabcde":
          factored_group1.append(final_string)  
       #break
       #print(b_factors)
   #print(chars_c)
   print(factored_group1) 

   adj_m = np.zeros((9,9),dtype=int)
   for mrf in factored_group1:
       adj_m += b.create_adjacency_matrix(mrf=mrf)

   adj_m[adj_m>0]=1

   print(adj_m)
 





   #adj_m[adj_m>0]=1

   #print(adj_m)

   

   #print(b.num_innerbox_0(n=5))
   #print(b.num_outerbox_0(n=5))
   #print(b.num_outerbox_0(n=4))
   #print(b.num_outerbox_0(n=3))
   #print(b.num_outerbox_0(n=2))


   #sys.stdout = codecs.lookup('utf-8')[-1](sys.stdout)

   #greek_letterz=[unichr(code) for code in range(945,970)]
   #print(greek_letterz)

   #b = BoxAnalysis()
   #b.zero_box_factor_4()
   #ar = b.zero_box_factor_5()
   #print(len(ar))
   
   #ZERO BOX FACTOR 4
   
   #chars_c = []
   #mrfs = b.read_file(name="4.txt")
   
   #b.zero_box_factor_5()


   #b_factors = []
   #factored_abc = []
   #factored_group1 = []
   #for mrf in mrfs:
   #    second_list,chars = b.zero_box_factor_4(s=mrf)
       #b_factors = b_factors+second_list
   #    b_factors = b_factors + list(set(second_list) - set(b_factors))
       #factored_abc.append(final_string)
   #    if not chars in chars_c:
   #       chars_c.append(chars)
       #if "".join(sorted(final_string)) == "ABCDabcde":
       #   factored_group1.append(final_string)
   #print(b_factors)
   #print(len(b_factors))
   #print(len(chars_c))
   #print(len(mrfs))
   '''

   #ZERO BOX FACTOR 3
   '''
   b.zero_box_factor()
   b_factors = []
   chars_c = []
   mrfs = b.read_file(name="3.txt")
   factored_abc = []
   factored_group1 = []
   for mrf in mrfs:
       second_list,chars,final_string = b.zero_box_factor(s=mrf)
       b_factors = b_factors + list(set(second_list) - set(b_factors))
       factored_abc.append(final_string)
       if not chars in chars_c:
          chars_c.append(chars)
       if "".join(sorted(final_string)) == "ABCDabcde":
          factored_group1.append(final_string)  
       #break
       #print(b_factors)
   #print(chars_c)
   print(factored_group1) 

   adj_m = np.zeros((9,9),dtype=int)
   for mrf in factored_group1:
       adj_m += b.create_adjacency_matrix(mrf=mrf)

   adj_m[adj_m>0]=1

   print(adj_m)

   import networkx as nx  
   import matplotlib.pyplot as plt



   G=nx.DiGraph()
   G.add_node(0),G.add_node(1),G.add_node(2),G.add_node(3),G.add_node(4),G.add_node(5),G.add_node(6),G.add_node(7),G.add_node(8)
   for row in range(9):
       for column in range(9):
           if adj_m[row,column] == 1:
              G.add_edge(row,column)                     
   nx.draw(G, pos=nx.circular_layout(G), with_labels=True, font_weight='bold')
   plt.show()
   '''
      
   #G.add_node(0),G.add_node(1),G.add_node(2),G.add_node(3),G.add_node(4)
   #G.add_edge(0, 1),G.add_edge(1, 2),G.add_edge(0, 2),G.add_edge(1, 4),G.add_edge(1, 3),G.add_edge(3, 2),G.add_edge(3,1),G.add_edge(4,3)
   #nx.draw(G, with_labels=True, font_weight='bold')
   #plt.show()

   
   #print("".join(sorted(factored_abc[0])))    
    
   
   #for mrf in mrfs:
   #    print(mrf)
       #print(b.factorize(s=mrf))
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
