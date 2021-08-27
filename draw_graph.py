import scipy.io
import pylab as plt
import math
import numpy as np
import difflib
import pickle
from matplotlib.patches import Ellipse
#from nltk import ngrams
from mpl_toolkits import mplot3d

#import numpy as np  
import scipy
#import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances #jaccard diss.
from sklearn import manifold  # multidimensional scaling
from sklearn.mixture import GaussianMixture

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyArrowPatch

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)


class DrawGraph():

      def __init__(self):
          pass

      def create_disctionaries(self):
          coord_color_cycles = {}
          coord_color_cycles["k"] = np.array([[-2,-2,0],[-2,-2,2],[-2,-2,4],[-2,-2,6],[-2,-2,8],[-2,-2,10]])
          coord_color_cycles["r"] = np.array([[-1,-1,0],[-1,-1,2],[-1,-1,4],[-1,-1,6],[-1,-1,8],[-1,-1,10]])
          coord_color_cycles["c"] = np.array([[-2,2,0],[-2,2,2],[-2,2,4],[-2,2,6],[-2,2,8],[-2,2,10]])
          coord_color_cycles["m"] = np.array([[2,-2,0],[2,-2,2],[2,-2,4],[2,-2,6],[2,-2,8],[2,-2,10]])
          coord_color_cycles["y"] = np.array([[-1,1,0],[-1,1,2],[-1,1,4],[-1,1,6],[-1,1,8],[-1,1,10]])
          coord_color_cycles["g"] = np.array([[1,-1,0],[1,-1,2],[1,-1,4],[1,-1,6],[1,-1,8],[1,-1,10]])
          coord_color_cycles["w"] = np.array([[1,1,0],[1,1,2],[1,1,4],[1,1,6],[1,1,8],[1,1,10]])
          coord_color_cycles["b"] = np.array([[2,2,0],[2,2,2],[2,2,4],[2,2,6],[2,2,8],[2,2,10]])
          
          up_or_down = {}
          up_or_down["k"] = 1 #up
          up_or_down["r"] = 0 #down
          up_or_down["c"] = 0
          up_or_down["m"] = 0
          up_or_down["y"] = 1
          up_or_down["g"] = 1
          up_or_down["w"] = 0
          up_or_down["b"] = 1
          
          connecting_tracks = {}
          connecting_tracks["km"] = np.array([2,0,1,2,0,1]) #1 - away from black, 2 - away from magenta, 0 - no connection
          connecting_tracks["kr"] = np.array([0,1,2,0,1,2])
          connecting_tracks["kc"] = np.array([1,2,0,1,2,0]) 
          connecting_tracks["rg"] = np.array([1,0,2,1,0,2]) 
          connecting_tracks["ry"] = np.array([2,1,0,2,1,0]) 
          connecting_tracks["mg"] = np.array([0,2,1,0,2,1])  
          connecting_tracks["mb"] = np.array([2,1,0,2,1,0])
          connecting_tracks["cy"] = np.array([0,2,1,0,2,1])
          connecting_tracks["cb"] = np.array([1,0,2,1,0,2])
          connecting_tracks["wg"] = np.array([2,1,0,2,1,0])
          connecting_tracks["wy"] = np.array([1,0,2,1,0,2])
          connecting_tracks["wb"] = np.array([0,2,1,0,2,1])
          
          labels = {}
          labels["k"] = ["0130","13021","0210","21032","0320","32013"]
          labels["r"] = ["2132","30213","2302","01230","2012","13201"]
          labels["y"] = ["1021","02130","1301","30123","1231","23102"]
          labels["b"] = ["2312","31203","2032","03210","2102","10231"]
          labels["w"] = ["0310","12031","0120","23012","0230","31023"]
          labels["c"] = ["3023","21302","3213","10321","3103","02310"] 
          labels["g"] = ["3203","20312","3123","12301","3013","01320"] 
          labels["m"] = ["1201","03120","1031","32103","1321","20132"]

          #print(coord_color_cycles)
          return coord_color_cycles,up_or_down,connecting_tracks,labels  

      def plot_3d_scatter(self):
          fig = plt.figure()
          ax = plt.axes(projection='3d')

          zdata = 15 * np.random.random(100)
	  xdata = np.sin(zdata) + 0.1 * np.random.randn(100)
          ydata = np.cos(zdata) + 0.1 * np.random.randn(100)
          ax.scatter3D(xdata, ydata, zdata);

          plt.show()

      def plot_3d_scatter_subgraph(self,c_col_cycles,up_or_down,connecting_tracks,labels):
          fig = plt.figure()
          ax = plt.axes(projection='3d')

          for key in c_col_cycles:
              data_points = c_col_cycles[key]
              if up_or_down[key] == 0:
                 data_points = data_points[::-1]
              k = 0
              #"->"
              for d in data_points:
                  ax.scatter3D(d[0], d[1], d[2],c=key,edgecolors='k',s=100)
                  if k <> len(data_points)-1:
                     arw = Arrow3D([d[0],data_points[k+1,0]],[d[1],data_points[k+1,1]],[d[2],data_points[k+1,2]], arrowstyle="-|>", color="black", lw = 1, mutation_scale=15)
                     ax.add_artist(arw)
                  k += 1

          for key in connecting_tracks:
              c1 = key[0]
              c2 = key[1]

              d1 = c_col_cycles[c1]
              d2 = c_col_cycles[c2]
              
              k = 0
              for arrow_type in connecting_tracks[key]:
                  if arrow_type == 1:
                     arw = Arrow3D([d1[k,0],d2[k,0]],[d1[k,1],d2[k,1]],[d1[k,2],d2[k,2]], arrowstyle="-|>", color="black", lw = 1, mutation_scale=15)
                     ax.add_artist(arw)
                  elif arrow_type == 2: 
                     arw = Arrow3D([d2[k,0],d1[k,0]],[d2[k,1],d1[k,1]],[d2[k,2],d1[k,2]], arrowstyle="-|>", color="black", lw = 1, mutation_scale=15)
                     ax.add_artist(arw)
                  k += 1   
          x = 0.1
          for key in labels:
              data_points = c_col_cycles[key]
              l = labels[key]
              
              for k in range(len(l)):
                  ax.text(data_points[k,0]+x/2, data_points[k,1]+x/2, data_points[k,2]+3*x, l[k], size=10, zorder=5, color='w',bbox={'facecolor': 'blue', 'alpha': 0.5, 'pad': 1}) 

          ax.xaxis.set_ticklabels([])
          ax.yaxis.set_ticklabels([])
          ax.zaxis.set_ticklabels([])

          for line in ax.xaxis.get_ticklines():
              line.set_visible(False)
          for line in ax.yaxis.get_ticklines():
              line.set_visible(False)
          for line in ax.zaxis.get_ticklines():
              line.set_visible(False)

          plt.show()  
  

 

if __name__ == "__main__":
   d = DrawGraph()
   c_col_cycles,up_or_down,connecting_tracks,labels  = d.create_disctionaries()
   d.plot_3d_scatter_subgraph(c_col_cycles,up_or_down,connecting_tracks,labels)
   
   #print("hallo")
